from __future__ import annotations

import warnings

import torch
import torch.nn as nn


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


class BoucWenRecurrentModel(nn.Module):
    """Black-box recurrent baseline. A small MLP maps the initial-
    condition vector y0 to the initial hidden state of an RNN/GRU/LSTM.

    `use_output_feedback=True` turns this into a NARX-style model that
    ALSO takes the previous output y(t-1) as an input at every step, in
    addition to u(t) -- specifically to support the benchmark's own
    explicit request (Noel & Schoukens 2016, Section 5) to report BOTH
    a "simulation" figure of merit (model driven only by u, autoregressive
    on its own past predictions) AND a "prediction" figure of merit
    (model driven by u AND the TRUE past output, i.e. teacher-forced
    one-step-ahead). With `use_output_feedback=False` (default), this
    model only supports simulation mode, like every other model in this
    project. See `forward(..., teacher_forcing_y=...)`.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, n_hidden_states=32, hidden_sizes=None,
                 num_layers=1, recurrent="LSTM", activation="ReLU", dropout_prob=0.0,
                 direct_feedthrough=False, rnn_nonlinearity="tanh", use_output_feedback=False, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32]
        self.n_hidden_states = n_hidden_states
        self.num_layers = num_layers
        self.recurrent_type = recurrent.upper()
        self.direct_feedthrough = direct_feedthrough
        self.use_output_feedback = use_output_feedback
        self.supports_prediction_mode = use_output_feedback

        self.initial_state_net = FeedforwardNetwork(n_states, hidden_sizes, n_hidden_states, activation, dropout_prob)

        recurrent_input_size = n_inputs + (n_outputs if use_output_feedback else 0)
        recurrent_key = self.recurrent_type
        common_args = {"input_size": recurrent_input_size, "hidden_size": n_hidden_states, "num_layers": num_layers,
                       "batch_first": True, "dropout": dropout_prob if num_layers > 1 else 0.0}
        if recurrent_key == "LSTM":
            self.recurrent = nn.LSTM(**common_args)
        elif recurrent_key == "GRU":
            self.recurrent = nn.GRU(**common_args)
        elif recurrent_key == "RNN":
            self.recurrent = nn.RNN(nonlinearity=rnn_nonlinearity.lower(), **common_args)
        else:
            raise ValueError("recurrent must be one of {'RNN', 'GRU', 'LSTM'}")
        self.output_layer = nn.Linear(n_hidden_states, n_outputs)
        self.direct_layer = nn.Linear(n_inputs, n_outputs) if direct_feedthrough else None

    def build_initial_state(self, y0):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        hidden_single = self.initial_state_net(y0)
        hidden = hidden_single.unsqueeze(0).repeat(self.num_layers, 1, 1)
        if self.recurrent_type == "LSTM":
            return hidden, hidden.clone()
        return hidden

    def forward(self, u: torch.Tensor, y0: torch.Tensor, teacher_forcing_y: torch.Tensor | None = None):
        """If `use_output_feedback=False`: `teacher_forcing_y` is ignored,
        this is a plain autoregressive (simulation-mode) recurrent model.

        If `use_output_feedback=True`:
          - `teacher_forcing_y=None` -> SIMULATION mode: at each step, the
            model's own previous prediction is fed back as the "past
            output" input (fully autoregressive, matching the paper's
            simulation-mode definition: y_mod(t) = F(u(1..t))).
          - `teacher_forcing_y=<tensor>` -> PREDICTION mode: the TRUE past
            output (shifted by one step) is fed back instead of the
            model's own prediction, matching the paper's prediction-mode
            definition exactly: y_mod(t) = F(u(1..t), y(1..t-1)).
        """
        initial_state = self.build_initial_state(y0)

        if not self.use_output_feedback:
            output, hidden_state = self.recurrent(u, initial_state)
            y_hat = self.output_layer(output)
            if self.direct_layer is not None:
                y_hat = y_hat + self.direct_layer(u)
            return y_hat, hidden_state

        batch_size, T, _ = u.shape
        device = u.device
        hidden_state = initial_state
        y0_flat = y0.squeeze(1) if y0.dim() == 3 else y0
        n_outputs = self.output_layer.out_features
        prev_y = y0_flat[:, -n_outputs:]

        outputs = []
        for t in range(T):
            u_t = u[:, t:t + 1, :]
            step_input = torch.cat([u_t, prev_y.unsqueeze(1)], dim=-1)
            step_out, hidden_state = self.recurrent(step_input, hidden_state)
            y_hat_t = self.output_layer(step_out)
            if self.direct_layer is not None:
                y_hat_t = y_hat_t + self.direct_layer(u_t)
            outputs.append(y_hat_t)
            if teacher_forcing_y is not None:
                prev_y = teacher_forcing_y[:, t, :]
            else:
                prev_y = y_hat_t.squeeze(1)

        y_hat = torch.cat(outputs, dim=1)
        return y_hat, hidden_state


def boucwen_rhs(state: torch.Tensor, u_t: torch.Tensor, mL: torch.Tensor, cL: torch.Tensor, kL: torch.Tensor,
                 alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """state = [y, ydot, z]. Returns d(state)/dt, matching prepare.py's own
    boucwen_rhs exactly (Noel & Schoukens 2016, eqs. 1-3; nu fixed at 1,
    so |z|^(nu-1)=1 and the |z|^nu term reduces to |z|):
        mL*y'' + kL*y + cL*y' + z = u
        z' = alpha*y' - beta*(gamma*|y'|*z + delta*y'*|z|)
    """
    y, ydot, z = state[:, 0], state[:, 1], state[:, 2]
    yddot = (u_t - kL * y - cL * ydot - z) / mL
    zdot = alpha * ydot - beta * (gamma * torch.abs(ydot) * z + delta * ydot * torch.abs(z))
    return torch.stack([ydot, yddot, zdot], dim=-1)


def _boucwen_rollout_impl(u: torch.Tensor, state_init: torch.Tensor, dt: torch.Tensor, num_substeps: int,
                           mL: torch.Tensor, cL: torch.Tensor, kL: torch.Tensor, alpha: torch.Tensor,
                           beta: torch.Tensor, gamma: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RK4 rollout, matching prepare.py's own simulate_boucwen_rk4 exactly
    (same 4-stage weights), just batched and differentiable. `_boucwen_rollout`
    below is this function JIT-compiled where possible, with a safe eager-mode
    fallback -- moves the T x num_substeps sequential loop out of plain
    Python/autograd bookkeeping into a scripted graph, speeding up the
    forward pass specifically (the dominant cost for this kind of model)."""
    T = u.shape[1]
    state = state_init
    outputs: list[torch.Tensor] = []
    for t in range(T):
        u_t = u[:, t, :].squeeze(-1)
        for _ in range(num_substeps):
            k1 = boucwen_rhs(state, u_t, mL, cL, kL, alpha, beta, gamma, delta)
            k2 = boucwen_rhs(state + 0.5 * dt * k1, u_t, mL, cL, kL, alpha, beta, gamma, delta)
            k3 = boucwen_rhs(state + 0.5 * dt * k2, u_t, mL, cL, kL, alpha, beta, gamma, delta)
            k4 = boucwen_rhs(state + dt * k3, u_t, mL, cL, kL, alpha, beta, gamma, delta)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        outputs.append(state[:, 0].unsqueeze(-1))
    y_hat = torch.stack(outputs, dim=1)
    return y_hat, state


try:
    _boucwen_rollout = torch.jit.script(_boucwen_rollout_impl)
except Exception as _jit_exc:  # noqa: BLE001 -- deliberately broad: any JIT failure should degrade gracefully
    warnings.warn(
        f"torch.jit.script compilation failed for BoucWenModel's rollout ({_jit_exc}); "
        f"falling back to eager mode. Model is still numerically correct, just slower."
    )
    _boucwen_rollout = _boucwen_rollout_impl


class BoucWenModel(nn.Module):
    """
    Genuine white-box model: directly integrates the true Bouc-Wen physics
    (Noel & Schoukens 2016, eqs. 1-3 -- verified directly against the paper,
    matches prepare.py's own generate_training_data exactly):

        mL*y'' + kL*y + cL*y' + z = u
        z' = alpha*y' - beta*(gamma*|y'|*|z|^(nu-1)*z + delta*y'*|z|^nu)

    States: [y, ydot, z] -- y is the measured displacement, z is the
    UNMEASURABLE hysteretic internal force whose own dynamics (eq. 3)
    are what actually encode the system's memory (the benchmark's own
    challenge #2). Integrated with fixed-substep RK4
    (`dt = Ts/num_substeps`, Ts = 1/750s -- NOT dt = 1/num_substeps,
    which decouples the integration step from the system's real
    timescale and causes severe instability).

    ALL 5 free physical parameters (mL, cL, kL, alpha, beta) are
    reference-scaled: stored as a learnable UNIT-SCALE multiplier of the
    paper's own Table 1 value (5 orders of magnitude apart, e.g. cL=10
    vs kL=5e4), so a single learning rate treats all of them comparably
    -- raw, unscaled values would badly starve the smaller-magnitude
    parameters' gradients. `gamma`/`delta` are already O(1) in the
    paper's own values (0.8, -1.1) and are learned directly, unscaled.
    `nu` is FIXED at 1, not learned: the paper's own Section 6 flags
    this exponent specifically as unusually hard ("the nonlinear
    functional form... is nonlinear in the parameter nu"); fixing it at
    the paper's own true value avoids that specific difficulty while
    still exercising the rest of the hysteretic identification
    challenge (the internal, unmeasurable `z` state itself).

    `requires_raw_io = True`: this model's parameters are calibrated to
    REAL physical (SI) scales, so it needs RAW, un-normalized u/y0 as
    input, unlike every black-box/structural model in this project --
    feeding it z-scored input while its physics stays at true SI scale
    causes a silent unit mismatch (the model's honest physical
    prediction becomes many orders of magnitude smaller than the
    normalized target, indistinguishable from an untrained all-zero
    output). `train.py`/`test.py` must check this flag and route raw
    (denormalized) `u`/`y0` to this model, then compare its prediction
    against the target in NORMALIZED units via
    `normalizer.normalize_y_tensor()` for a fair, comparable
    `validation_rmse_norm_mean` against every other model type.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, num_substeps=5, hidden_sizes=None,
                 activation="Tanh", dropout_prob=0.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        self.num_substeps = num_substeps
        self.requires_raw_io = True

        self.state_init = FeedforwardNetwork(n_states, hidden_sizes, 3, activation=activation, dropout_prob=dropout_prob)
        nn.init.zeros_(self.state_init.net[-1].weight)
        nn.init.zeros_(self.state_init.net[-1].bias)

        # Reference scales matching the paper's own Table 1 exactly
        # (verified directly against the paper this session).
        self.mL_ref, self.cL_ref, self.kL_ref = 2.0, 10.0, 5.0e4
        self.alpha_ref, self.beta_ref = 5.0e4, 1.0e3
        self.mL_scale = nn.Parameter(torch.tensor(1.0))
        self.cL_scale = nn.Parameter(torch.tensor(1.0))
        self.kL_scale = nn.Parameter(torch.tensor(1.0))
        self.alpha_scale = nn.Parameter(torch.tensor(1.0))
        self.beta_scale = nn.Parameter(torch.tensor(1.0))
        self.gamma = nn.Parameter(torch.tensor(0.8))
        self.delta = nn.Parameter(torch.tensor(-1.1))
        self.nu = 1.0  # fixed, not learned -- see class docstring

    def forward(self, u: torch.Tensor, y0: torch.Tensor, sampling_time: float = 1.0 / 750.0) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        state = self.state_init(y0)  # (batch, 3): [y, ydot, z]

        mL = self.mL_ref * self.mL_scale
        cL = self.cL_ref * self.cL_scale
        kL = self.kL_ref * self.kL_scale
        alpha = self.alpha_ref * self.alpha_scale
        beta = self.beta_ref * self.beta_scale

        dt = torch.tensor(sampling_time / self.num_substeps, dtype=u.dtype, device=u.device)
        y_hat, final_state = _boucwen_rollout(u, state, dt, self.num_substeps, mL, cL, kL, alpha, beta, self.gamma, self.delta)
        return y_hat, final_state


def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return BoucWenRecurrentModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"], hidden_sizes=hidden_sizes,
            num_layers=config_pars["num_layers"], recurrent=model_type, activation=activation,
            dropout_prob=dropout_prob, direct_feedthrough=config_pars.get("direct_feedthrough", False),
            rnn_nonlinearity=config_pars.get("rnn_nonlinearity", "tanh"),
            use_output_feedback=config_pars.get("use_output_feedback", False),
        )

    if model_type == "BOUCWEN":
        return BoucWenModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            num_substeps=config_pars.get("num_substeps", 5), hidden_sizes=hidden_sizes,
            activation=activation, dropout_prob=dropout_prob,
        )

    raise ValueError("Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'BOUCWEN'}.")