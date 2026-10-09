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


class PWHRecurrentModel(nn.Module):
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


class ParallelWHModel(nn.Module):
    """
    Grey-box model matching this benchmark's actual physical structure
    (Schoukens et al. 2017): `n_branches` parallel Wiener-Hammerstein
    branches sharing the same input, outputs summed:

        y(k) = sum_i S[i](q) f[i](H[i](q) u(k))

    where H[i]/S[i] are the front/back LTI blocks of branch i (3rd order
    in the real device under test) and f[i] is a static nonlinearity
    (a diode-resistor network in the real device). The real system has
    2 branches (`n_branches=2` default).

    H[i]/S[i] are each implemented as a learnable causal FIR filter
    (`FIRBlock`, a single `conv1d` -- fully vectorized, no sequential
    Python loop) rather than a recursive IIR state-space realization.
    This is a deliberate choice, consistent with the WienerHammerBenchMark
    project and directly informed by the BoucWen/Silverbox projects'
    experience: a sequential per-timestep recursion for a genuinely
    nonlinear/stateful system was found fragile (numerical instability
    issues needing careful sign/saturation fixes) and slow to backprop
    through (a deep sequential graph dominates training time). An FIR
    approximation of the (here, purely LTI) H[i]/S[i] blocks sidesteps
    both problems entirely, and -- as a bonus -- needs no initial-
    condition estimation at all, since an FIR filter's only "memory" is
    its own finite tap length, fully contained within any training window
    (the `forward(u, y0)` signature is kept for interface consistency;
    y0 is unused here).

    `f[i]` is a small pointwise MLP per branch (applied independently at
    each timestep, memoryless by construction, matching the true diode
    circuit). `n_taps_h`/`n_taps_s` trade approximation quality for the
    true 3rd-order IIR responses against model size/training cost.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_branches=2, n_taps_h=64, n_taps_s=64,
                 hidden_sizes=None, activation="Tanh", dropout_prob=0.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [16]

        self.n_branches = n_branches
        self.front_filters = nn.ModuleList([FIRBlock(n_taps=n_taps_h, n_inputs=n_inputs, n_outputs=1) for _ in range(n_branches)])
        self.nonlinearities = nn.ModuleList([
            FeedforwardNetwork(1, hidden_sizes, 1, activation=activation, dropout_prob=dropout_prob) for _ in range(n_branches)
        ])
        self.back_filters = nn.ModuleList([FIRBlock(n_taps=n_taps_s, n_inputs=1, n_outputs=n_outputs) for _ in range(n_branches)])

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, None]:
        y_hat = None
        for branch_index in range(self.n_branches):
            x = self.front_filters[branch_index](u)              # (batch, T, 1)
            r = self.nonlinearities[branch_index](x)              # pointwise, (batch, T, 1)
            branch_output = self.back_filters[branch_index](r)    # (batch, T, n_outputs)
            y_hat = branch_output if y_hat is None else y_hat + branch_output
        return y_hat, None


def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return PWHRecurrentModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"], hidden_sizes=hidden_sizes,
            num_layers=config_pars["num_layers"], recurrent=model_type, activation=activation,
            dropout_prob=dropout_prob, direct_feedthrough=config_pars.get("direct_feedthrough", False),
            rnn_nonlinearity=config_pars.get("rnn_nonlinearity", "tanh"),
        )

    if model_type == "PARALLELWH":
        return ParallelWHModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_branches=config_pars.get("n_branches", 2),
            n_taps_h=config_pars.get("n_taps_h", 64), n_taps_s=config_pars.get("n_taps_s", 64),
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
        )

    raise ValueError("Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'PARALLELWH'}.")