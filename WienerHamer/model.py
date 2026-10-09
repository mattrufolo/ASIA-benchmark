from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str) -> nn.Module:
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    if key == "sigmoid":
        return nn.Sigmoid()
    warnings.warn(f"Unknown activation `{name}`. Falling back to ReLU.")
    return nn.ReLU()


class FeedforwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="ReLU", dropout_prob=0.0):
        super().__init__()
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one element")
        layers: list[nn.Module] = []
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(make_activation(activation))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout(dropout_prob))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecurrentBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, recurrent="LSTM", dropout_prob=0.0, rnn_nonlinearity="tanh"):
        super().__init__()
        recurrent_key = recurrent.upper()
        common_args = {"input_size": input_size, "hidden_size": hidden_size, "num_layers": num_layers,
                       "batch_first": True, "dropout": dropout_prob if num_layers > 1 else 0.0}
        if recurrent_key == "LSTM":
            self.recurrent = nn.LSTM(**common_args)
        elif recurrent_key == "GRU":
            self.recurrent = nn.GRU(**common_args)
        elif recurrent_key == "RNN":
            self.recurrent = nn.RNN(nonlinearity=rnn_nonlinearity.lower(), **common_args)
        else:
            raise ValueError("recurrent must be one of {'RNN', 'GRU', 'LSTM'}")
        self.recurrent_type = recurrent_key
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_state=None):
        output, hidden_state = self.recurrent(x, hidden_state)
        return self.output_layer(output), hidden_state


class WHRecurrentModel(nn.Module):
    """Generic black-box recurrent model: a small MLP maps the
    initial-condition vector y0 to the initial hidden state of an
    RNN/GRU/LSTM that then rolls forward over u."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_hidden_states=32, hidden_sizes=None,
                 num_layers=1, recurrent="LSTM", activation="ReLU", dropout_prob=0.0,
                 direct_feedthrough=False, rnn_nonlinearity="tanh", **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32]
        self.n_hidden_states = n_hidden_states
        self.num_layers = num_layers
        self.recurrent_type = recurrent.upper()
        self.direct_feedthrough = direct_feedthrough

        self.initial_state_net = FeedforwardNetwork(n_states, hidden_sizes, n_hidden_states, activation, dropout_prob)
        self.recurrent_block = RecurrentBlock(n_inputs, n_hidden_states, n_outputs, num_layers, recurrent, dropout_prob, rnn_nonlinearity)
        self.direct_layer = nn.Linear(n_inputs, n_outputs) if direct_feedthrough else None

    def build_initial_state(self, y0):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        hidden_single = self.initial_state_net(y0)
        hidden = hidden_single.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.recurrent_type == "LSTM":
            return hidden, hidden.clone()
        return hidden

    def forward(self, u, y0):
        initial_state = self.build_initial_state(y0)
        y_hat, hidden_state = self.recurrent_block(u, initial_state)
        if self.direct_layer is not None:
            y_hat = y_hat + self.direct_layer(u)
        return y_hat, hidden_state


class FIRBlock(nn.Module):
    """A causal FIR (finite impulse response) filter, implemented as a
    single 1D convolution -- fully vectorized (no sequential Python loop),
    unlike an IIR/state-space recursion. Padded with `n_taps-1` ZEROS on
    the left of each window (a simplification: the true past input isn't
    reused here, since a window can start anywhere in the recording and
    plumbing the exact preceding samples through isn't necessary for an
    FIR block -- windowed random-crop training already means the model
    must be robust to arbitrary window starts anyway)."""

    def __init__(self, n_taps: int, n_inputs: int = 1, n_outputs: int = 1) -> None:
        super().__init__()
        self.n_taps = n_taps
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size=n_taps, bias=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_t = u.transpose(1, 2)  # (batch, n_inputs, T)
        u_padded = F.pad(u_t, (self.n_taps - 1, 0))
        y = self.conv(u_padded)  # (batch, n_outputs, T)
        return y.transpose(1, 2)  # (batch, T, n_outputs)


class WienerHammersteinModel(nn.Module):
    """Grey-box model matching this benchmark's actual physical structure
    (Schoukens, Suykens & Ljung, 2009): G1(LTI) -> f(.) (static
    nonlinearity) -> G2(LTI), the classic Wiener-Hammerstein block-oriented
    structure. G1/G2 are learnable causal FIR filters (see FIRBlock);
    `f` is a small pointwise MLP (applied independently at each timestep,
    since it's memoryless by construction -- matching the report's own
    description of the true nonlinearity as a static diode circuit).

    Unlike every other grey-box model in this ASIA collection so far
    (CED/EMPS/BoucWen), this one needs NO initial-condition estimation at
    all: an FIR filter's only "memory" is its own finite tap length, which
    is fully contained within the window itself (via the zero-padding in
    FIRBlock) -- there's no persistent hidden state to estimate from y0.
    The `forward(u, y0)` signature is kept for interface consistency with
    train.py, but y0 is simply unused here.

    `n_taps_g1`/`n_taps_g2` trade approximation quality for a true 3rd-
    order IIR filter (which can have an arbitrarily long impulse response)
    against model size/training cost -- the report's own G1 (3rd-order
    Chebyshev, 4.4kHz cutoff) and G2 (3rd-order inverse Chebyshev with a
    transmission zero near 5kHz) at fs~51.2kHz likely need on the order of
    tens to a few hundred taps for a reasonable approximation; this is
    worth tuning empirically.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_taps_g1=64, n_taps_g2=64,
                 hidden_sizes=None, activation="Tanh", dropout_prob=0.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [16]

        self.g1 = FIRBlock(n_taps=n_taps_g1, n_inputs=n_inputs, n_outputs=1)
        self.nonlin = FeedforwardNetwork(1, hidden_sizes, 1, activation=activation, dropout_prob=dropout_prob)
        self.g2 = FIRBlock(n_taps=n_taps_g2, n_inputs=1, n_outputs=n_outputs)

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, None]:
        x1 = self.g1(u)          # (batch, T, 1)
        x2 = self.nonlin(x1)     # pointwise, (batch, T, 1)
        y_hat = self.g2(x2)      # (batch, T, n_outputs)
        return y_hat, None


class WienerHammersteinFeedthroughModel(nn.Module):
    """Same G1 -> f -> G2 structure as WienerHammersteinModel, plus a
    learnable direct linear feedthrough term D*u added straight onto the
    output: `y_hat = G2(f(G1(u))) + D*u`. Motivated by the possibility of
    a parasitic/direct signal path in the real circuit alongside the
    dominant G1->f->G2 path (a common real-world deviation from the ideal
    block-oriented structure). Unlike WienerHammersteinHybridModel's
    residual branch, D is a single linear map (no extra nonlinearity/
    capacity), so this tests a structural addition rather than a capacity
    increase."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_taps_g1=64, n_taps_g2=64,
                 hidden_sizes=None, activation="Tanh", dropout_prob=0.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [16]

        self.g1 = FIRBlock(n_taps=n_taps_g1, n_inputs=n_inputs, n_outputs=1)
        self.nonlin = FeedforwardNetwork(1, hidden_sizes, 1, activation=activation, dropout_prob=dropout_prob)
        self.g2 = FIRBlock(n_taps=n_taps_g2, n_inputs=1, n_outputs=n_outputs)
        self.direct_layer = nn.Linear(n_inputs, n_outputs, bias=False)

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, None]:
        x1 = self.g1(u)
        x2 = self.nonlin(x1)
        y_hat = self.g2(x2) + self.direct_layer(u)
        return y_hat, None


class WienerHammersteinHybridModel(nn.Module):
    """Grey-box + black-box hybrid: the physical G1 -> f -> G2 FIR path
    (identical to `WienerHammersteinModel`) runs in parallel with a small
    recurrent residual branch that reads `[u, x2]` (raw input plus the
    post-nonlinearity signal) and whose initial hidden state is estimated
    from y0 exactly as in `WHRecurrentModel`. Final output is
    `y_phys + y_res`. Intent: let the residual branch pick up whatever
    the finite-tap FIR/static-nonlinearity approximation structurally
    cannot represent (e.g. the truncated tail of G1/G2's true IIR
    response, or the transmission zero's sharp phase behavior) without
    relearning the whole input-output map from scratch."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_taps_g1=64, n_taps_g2=64,
                 hidden_sizes=None, activation="Tanh", dropout_prob=0.0,
                 residual_hidden_states=24, residual_init_hidden_sizes=None,
                 residual_recurrent="GRU", **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [16]
        if residual_init_hidden_sizes is None:
            residual_init_hidden_sizes = [24]

        self.g1 = FIRBlock(n_taps=n_taps_g1, n_inputs=n_inputs, n_outputs=1)
        self.nonlin = FeedforwardNetwork(1, hidden_sizes, 1, activation=activation, dropout_prob=dropout_prob)
        self.g2 = FIRBlock(n_taps=n_taps_g2, n_inputs=1, n_outputs=n_outputs)

        self.residual_hidden_states = residual_hidden_states
        self.residual_init_state_net = FeedforwardNetwork(
            n_states, residual_init_hidden_sizes, residual_hidden_states, activation, dropout_prob,
        )
        self.residual_block = RecurrentBlock(
            n_inputs + 1, residual_hidden_states, n_outputs, num_layers=1,
            recurrent=residual_recurrent, dropout_prob=dropout_prob,
        )
        self.residual_recurrent_type = residual_recurrent.upper()

    def build_residual_initial_state(self, y0):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        hidden = self.residual_init_state_net(y0).unsqueeze(0)
        if self.residual_recurrent_type == "LSTM":
            return hidden, hidden.clone()
        return hidden

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, None]:
        x1 = self.g1(u)                              # (batch, T, 1)
        x2 = self.nonlin(x1)                          # (batch, T, 1)
        y_phys = self.g2(x2)                           # (batch, T, n_outputs)

        residual_input = torch.cat([u, x2], dim=-1)    # (batch, T, n_inputs + 1)
        residual_init_state = self.build_residual_initial_state(y0)
        y_res, _ = self.residual_block(residual_input, residual_init_state)

        return y_phys + y_res, None


def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return WHRecurrentModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"], hidden_sizes=hidden_sizes,
            num_layers=config_pars["num_layers"], recurrent=model_type, activation=activation,
            dropout_prob=dropout_prob, direct_feedthrough=config_pars.get("direct_feedthrough", False),
            rnn_nonlinearity=config_pars.get("rnn_nonlinearity", "tanh"),
        )

    if model_type == "WIENERHAMMERSTEIN":
        return WienerHammersteinModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_taps_g1=config_pars.get("n_taps_g1", 64), n_taps_g2=config_pars.get("n_taps_g2", 64),
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
        )

    if model_type == "WHHYBRID":
        return WienerHammersteinHybridModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_taps_g1=config_pars.get("n_taps_g1", 64), n_taps_g2=config_pars.get("n_taps_g2", 64),
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
            residual_hidden_states=config_pars.get("residual_hidden_states", 24),
            residual_init_hidden_sizes=config_pars.get("residual_init_hidden_sizes"),
            residual_recurrent=config_pars.get("residual_recurrent", "GRU"),
        )

    if model_type == "WIENERHAMMERSTEIN_FEEDTHROUGH":
        return WienerHammersteinFeedthroughModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_taps_g1=config_pars.get("n_taps_g1", 64), n_taps_g2=config_pars.get("n_taps_g2", 64),
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
        )

    raise ValueError(
        "Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'WIENERHAMMERSTEIN', "
        "'WHHYBRID', 'WIENERHAMMERSTEIN_FEEDTHROUGH'}."
    )