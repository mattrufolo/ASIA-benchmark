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


class WHPRecurrentModel(nn.Module):
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
    """A causal FIR filter, implemented as a single 1D convolution --
    fully vectorized (no sequential Python loop), unlike an IIR/state-space
    recursion. Padded with `n_taps-1` zeros on the left of each window."""

    def __init__(self, n_taps: int, n_inputs: int = 1, n_outputs: int = 1) -> None:
        super().__init__()
        self.n_taps = n_taps
        self.conv = nn.Conv1d(n_inputs, n_outputs, kernel_size=n_taps, bias=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u_t = u.transpose(1, 2)  # (batch, n_inputs, T)
        u_padded = F.pad(u_t, (self.n_taps - 1, 0))
        y = self.conv(u_padded)  # (batch, n_outputs, T)
        return y.transpose(1, 2)  # (batch, T, n_outputs)


class WienerHammersteinProcessNoiseModel(nn.Module):
    """
    Grey-box model matching this benchmark's physical structure directly
    (Schoukens & Noel, 2016/2017 -- confirmed from the paper's own Fig. 2
    and Section 4.2):

        R(s) -> [+ process noise e_x(t)] -> f(x) -> S(s) -> y

    R (front LTI) and S (back LTI, has a transmission zero -- the same
    identification challenge the plain WienerHammerBenchMark project's
    back filter has) are each a learnable causal FIR filter (`FIRBlock`,
    a single `conv1d` -- fully vectorized, no sequential Python loop).
    f[.] is a small pointwise MLP (the diode-resistor static nonlinearity).
    This reuses the exact FIR-based design already proven on the plain
    WienerHammerBenchMark project, for the same reasons: no
    initial-condition estimation is needed at all (an FIR filter's only
    "memory" is its own finite tap length, fully contained within any
    training window), and it avoids the sequential-loop numerical-
    stability/speed issues found on other projects in this collection
    (BoucWen, Silverbox, EMPS) that needed an actual per-timestep
    recursion. `y0` is kept in the signature for interface consistency
    but unused here.

    IMPORTANT DIFFERENCE FROM THE PLAIN WienerHammerBenchMark PROJECT:
    this benchmark's DOMINANT noise source is PROCESS noise entering
    BEFORE the nonlinearity (see class module docstring / description.md
    Section 1) -- not just small measurement noise. This model, being
    purely deterministic (R -> f -> S with no explicit noise term), can
    only ever learn the AVERAGE/expected mapping u -> y across the
    process-noise realizations it was trained on; it structurally cannot
    represent the specific noise realization present in any one training
    sequence. This is expected and appropriate: the benchmark's own test
    data is NOISELESS specifically so a correctly-identified deterministic
    backbone is what's being evaluated, not noise-realization matching.
    `n_taps_r`/`n_taps_s` trade FIR approximation quality for the true
    LTI responses against model size/training cost.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_taps_r=64, n_taps_s=64,
                 hidden_sizes=None, activation="Tanh", dropout_prob=0.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [16]

        self.front_filter = FIRBlock(n_taps=n_taps_r, n_inputs=n_inputs, n_outputs=1)
        self.nonlinearity = FeedforwardNetwork(1, hidden_sizes, 1, activation=activation, dropout_prob=dropout_prob)
        self.back_filter = FIRBlock(n_taps=n_taps_s, n_inputs=1, n_outputs=n_outputs)

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, None]:
        x = self.front_filter(u)         # (batch, T, 1)
        r = self.nonlinearity(x)          # pointwise, (batch, T, 1)
        y_hat = self.back_filter(r)       # (batch, T, n_outputs)
        return y_hat, None


def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return WHPRecurrentModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"], hidden_sizes=hidden_sizes,
            num_layers=config_pars["num_layers"], recurrent=model_type, activation=activation,
            dropout_prob=dropout_prob, direct_feedthrough=config_pars.get("direct_feedthrough", False),
            rnn_nonlinearity=config_pars.get("rnn_nonlinearity", "tanh"),
        )

    if model_type == "WIENERHAMMERSTEIN":
        return WienerHammersteinProcessNoiseModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_taps_r=config_pars.get("n_taps_r", 64), n_taps_s=config_pars.get("n_taps_s", 64),
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
        )

    raise ValueError("Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'WIENERHAMMERSTEIN'}.")