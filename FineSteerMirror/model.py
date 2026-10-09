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


class FSMRecurrentModel(nn.Module):
    """Black-box MIMO recurrent baseline (3 inputs, 3 outputs). A small
    MLP maps the initial-condition vector y0 to the initial hidden state
    of an RNN/GRU/LSTM that then rolls forward over the 3-channel input,
    producing all 3 output channels jointly."""

    def __init__(self, n_inputs=3, n_states=1, n_outputs=3, n_hidden_states=64, hidden_sizes=None,
                 num_layers=1, recurrent="LSTM", activation="ReLU", dropout_prob=0.0,
                 direct_feedthrough=False, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64]
        self.n_hidden_states = n_hidden_states
        self.num_layers = num_layers
        self.recurrent_type = recurrent.upper()
        self.direct_feedthrough = direct_feedthrough

        self.initial_state_net = FeedforwardNetwork(n_states, hidden_sizes, n_hidden_states, activation, dropout_prob)

        common_args = {"input_size": n_inputs, "hidden_size": n_hidden_states, "num_layers": num_layers,
                       "batch_first": True, "dropout": dropout_prob if num_layers > 1 else 0.0}
        if self.recurrent_type == "LSTM":
            self.recurrent = nn.LSTM(**common_args)
        elif self.recurrent_type == "GRU":
            self.recurrent = nn.GRU(**common_args)
        elif self.recurrent_type == "RNN":
            self.recurrent = nn.RNN(**common_args)
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

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, object]:
        initial_state = self.build_initial_state(y0)
        output, hidden_state = self.recurrent(u, initial_state)
        y_hat = self.output_layer(output)
        if self.direct_layer is not None:
            y_hat = y_hat + self.direct_layer(u)
        return y_hat, hidden_state


class FSMHysteresisLFRModel(nn.Module):
    """
    Genuine white-box model, directly reflecting the benchmark's own
    confirmed physics -- now grounded in the actual paper and companion
    presentation (Floren et al., ISMA-USD2024; "Dataset and baseline for
    the CubeSpec Fine Steering Mirror", NLB Workshop 2026 -- both read
    directly this session):

    "the system behaves mostly linearly, but the presence of hysteresis
    in the piezo-actuators introduces dynamic nonlinearities" (official
    benchmark page). Confirmed directly from the paper: fs=6400Hz,
    excitation up to fmax=3000Hz, with the most dominant resonance peaks
    concentrated in the **750-950Hz** band. Critically, the paper's own
    identified 28th-order linear (BLA) model contains **two real poles**
    (i.e. non-oscillatory, zero-frequency), which the authors explicitly
    attribute to the piezo-actuators' hysteresis: "This conjecture is
    substantiated by the fact that the obtained model contains two real
    poles... This is consistent with the general definition of a
    hysteretic system, in which the hysteresis loop persists as the
    input frequency approaches zero." This directly motivates including
    both oscillatory AND real (non-oscillatory) states below, not just
    oscillatory ones.

    The paper's own "next step" (presentation, slide 18) is an NL-LFR
    model:

        x(n+1) = A x(n) + Bu u(n) + Bw w(n)
        y(n)   = Cy x(n) + Dyu u(n) + Dyw w(n)
        z(n)   = Cz x(n) + Dzu u(n)
        w(n)   = f(z(n))

    -- a linear core (A, Bu, Cy, Dyu, identical to their BLA model) with
    an additional nonlinear feedback loop (a *learned feedforward neural
    network* f, nz=16, nw=8, 5768 total parameters, trained on all 3
    amplitude levels combined -- NRMSE 3.5-7%, versus 4.5-42% for the
    plain linear model depending on train/test amplitude match).

    This project's own structure differs deliberately: rather than a
    generic NN feedback loop from state to state, the nonlinearity is
    placed explicitly at the INPUT, one independent Bouc-Wen-style
    hysteresis operator per piezo-actuator -- since piezo-actuators are
    themselves the confirmed, physically-specific source of the
    hysteresis (the paper's own stated hypothesis), input-referred
    hysteresis is arguably a more directly interpretable placement than
    an abstract state-feedback loop, while still being structurally
    close to the same LFR spirit (linear dynamics + an explicit
    nonlinear correction).

    Structure:
      1. THREE INDEPENDENT Bouc-Wen-style hysteresis operators, one per
         input channel:

            z_i' = alpha_i*u_i' - beta_i*(gamma_i*|u_i'|*z_i + delta_i*u_i'*|z_i|)   (nu=1)
            u_corrected_i = c_gain_i * u_i + z_i

      2. `n_real_modes` (default 2, matching the paper's own confirmed
         finding) independent REAL (non-oscillatory) states, each a
         simple stable first-order decay, PLUS `n_complex_modes`
         (default 3) oscillatory 2x2 modal blocks -- decay-rate/softplus
         parameterized (guaranteed stable for ANY raw parameter value,
         verified directly up to and beyond fmax=3000Hz at fs=6400Hz --
         see the note further down about why the a_coef/b_coef formula
         had to be RE-DERIVED, not reused, from this collection's other
         white-box models). Each mode (real or complex) is driven by a
         learnable 3-vector combination of the 3 hysteresis-corrected
         input channels.

      3. The 3 measured displacements are a learnable linear combination
         of ALL modal states (real + complex), matching the "three
         non-collocated reference points" measurement setup.

    `pole_freq_hz_init` defaults to a spread centered on the paper's own
    confirmed dominant peak band (750-950Hz), rather than an arbitrary
    generic guess -- a genuinely informed starting point, though the
    exact frequencies/damping are still for `train.py`'s search to tune,
    not something to trust blindly.

    `state_init`'s output layer is zero-initialized (same convention
    used throughout this collection).
    """

    def __init__(self, n_inputs=3, n_states=1, n_outputs=3, n_complex_modes=3, n_real_modes=2,
                 hidden_sizes=None, activation="Tanh", dropout_prob=0.0, sampling_time=1.0 / 6400.0,
                 e_pos=0.05, pole_freq_hz_init=None, pole_damping_ratio_init=0.02,
                 real_decay_hz_init=None, **kwargs) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if n_complex_modes < 0 or n_real_modes < 0 or n_complex_modes + n_real_modes < 1:
            raise ValueError("Need at least one mode total (n_complex_modes + n_real_modes >= 1).")
        if n_inputs != 3 or n_outputs != 3:
            raise ValueError("FSMHysteresisLFRModel is specific to the 3-input, 3-output FSM benchmark.")

        self.n_complex_modes = n_complex_modes
        self.n_real_modes = n_real_modes
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.dt = float(sampling_time)
        self.e_pos = e_pos  # fixed constant, same reasoning as Silverbox's own model

        n_total_modal_states = 2 * n_complex_modes + n_real_modes
        self.state_init = FeedforwardNetwork(n_states, hidden_sizes, n_total_modal_states + 3, activation=activation, dropout_prob=dropout_prob)
        nn.init.zeros_(self.state_init.net[-1].weight)
        nn.init.zeros_(self.state_init.net[-1].bias)

        # Complex (oscillatory) modes -- default spread centered on the
        # paper's own confirmed dominant peak band, 750-950Hz.
        if n_complex_modes > 0:
            if pole_freq_hz_init is None:
                if n_complex_modes == 1:
                    pole_freq_hz_init = [850.0]
                else:
                    pole_freq_hz_init = [750.0 + i * (200.0 / max(n_complex_modes - 1, 1)) for i in range(n_complex_modes)]
            if len(pole_freq_hz_init) != n_complex_modes:
                raise ValueError("pole_freq_hz_init must have exactly n_complex_modes entries.")
            omega_init_rad = torch.tensor([2 * 3.141592653589793 * f for f in pole_freq_hz_init], dtype=torch.float32)
            decay_rate_init = float(pole_damping_ratio_init) * omega_init_rad
            self.complex_decay_rate_raw = nn.Parameter(torch.log(torch.expm1(decay_rate_init)))
            self.pole_angle = nn.Parameter(omega_init_rad * self.dt)
            self.c_participation_complex = nn.Parameter(torch.randn(n_complex_modes, 3) * 0.1 + 1.0 / max(n_complex_modes, 1))
            self.d_output_complex = nn.Parameter(torch.randn(n_outputs, n_complex_modes) * 0.1 + 1.0 / max(n_complex_modes, 1))

        # Real (non-oscillatory) modes -- directly motivated by the
        # paper's own finding of two real poles in the identified linear
        # model, attributed to piezo-actuator hysteresis. A generic,
        # modest default decay rate (not benchmark-specific-tuned) --
        # finding good values is exactly what train.py's search is for.
        if n_real_modes > 0:
            if real_decay_hz_init is None:
                real_decay_hz_init = [50.0] * n_real_modes
            if len(real_decay_hz_init) != n_real_modes:
                raise ValueError("real_decay_hz_init must have exactly n_real_modes entries.")
            real_decay_rate_init = torch.tensor([2 * 3.141592653589793 * f for f in real_decay_hz_init], dtype=torch.float32)
            self.real_decay_rate_raw = nn.Parameter(torch.log(torch.expm1(real_decay_rate_init)))
            self.c_participation_real = nn.Parameter(torch.randn(n_real_modes, 3) * 0.1 + 1.0 / max(n_real_modes, 1))
            self.d_output_real = nn.Parameter(torch.randn(n_outputs, n_real_modes) * 0.1 + 1.0 / max(n_real_modes, 1))

        # Per-channel Bouc-Wen-style hysteresis parameters (nu=1).
        self.alpha = nn.Parameter(torch.full((3,), 1.0))
        self.beta = nn.Parameter(torch.full((3,), 0.5))
        self.gamma = nn.Parameter(torch.full((3,), 0.8))
        self.delta = nn.Parameter(torch.full((3,), -0.3))
        self.c_gain = nn.Parameter(torch.ones(3))

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        state0 = self.state_init(y0)
        idx = 0
        if self.n_complex_modes > 0:
            q = state0[:, idx: idx + self.n_complex_modes]; idx += self.n_complex_modes
            qdot = state0[:, idx: idx + self.n_complex_modes]; idx += self.n_complex_modes
        if self.n_real_modes > 0:
            r = state0[:, idx: idx + self.n_real_modes]; idx += self.n_real_modes
        z = state0[:, idx:]

        if self.n_complex_modes > 0:
            magnitude = torch.exp(-nn.functional.softplus(self.complex_decay_rate_raw) * self.dt)
        if self.n_real_modes > 0:
            real_magnitude = torch.exp(-nn.functional.softplus(self.real_decay_rate_raw) * self.dt)  # (n_real_modes,), in (0,1)

        outputs = []
        u_prev = u[:, 0, :]
        for t in range(T):
            u_t = u[:, t, :]
            udot = (u_t - u_prev) / self.dt if t > 0 else torch.zeros_like(u_t)
            u_prev = u_t

            zdot = self.alpha.unsqueeze(0) * udot - self.beta.unsqueeze(0) * (
                self.gamma.unsqueeze(0) * torch.abs(udot) * z + self.delta.unsqueeze(0) * udot * torch.abs(z)
            )
            z = z + self.dt * zdot
            u_corrected = self.c_gain.unsqueeze(0) * u_t + z  # (batch, 3)

            y_t = 0.0

            if self.n_complex_modes > 0:
                # Semi-implicit (symplectic) Euler; a_coef/b_coef derived
                # from the ACTUAL closed-loop transition matrix of this
                # update order (det=a, trace=a+e_pos*b+1) -- NOT the
                # formula used for a plain (non-substituted) recursion
                # elsewhere in this collection. Re-derived and verified
                # directly (symbolically and numerically) after finding
                # that reusing the other formula here is only stable at
                # low frequencies and genuinely unstable above ~200Hz --
                # this benchmark's confirmed 750-950Hz dominant peaks
                # would have silently hit that bug.
                drive_c = torch.matmul(u_corrected, self.c_participation_complex.t())
                det = (magnitude ** 2).unsqueeze(0)
                trace = 2 * magnitude.unsqueeze(0) * torch.cos(self.pole_angle).unsqueeze(0)
                a_coef = det
                b_coef = (trace - a_coef - 1.0) / self.e_pos
                qdot = a_coef * qdot + b_coef * q + drive_c
                q = q + self.e_pos * qdot
                y_t = y_t + torch.matmul(q, self.d_output_complex.t())

            if self.n_real_modes > 0:
                # Simple stable first-order decay per real mode --
                # magnitude in (0,1) guaranteed by construction, matching
                # the paper's own confirmed real (zero-frequency) poles.
                drive_r = torch.matmul(u_corrected, self.c_participation_real.t())
                r = real_magnitude.unsqueeze(0) * r + self.dt * drive_r
                y_t = y_t + torch.matmul(r, self.d_output_real.t())

            outputs.append(y_t.unsqueeze(1))

        y_hat = torch.cat(outputs, dim=1)
        state_parts = []
        if self.n_complex_modes > 0:
            state_parts += [q, qdot]
        if self.n_real_modes > 0:
            state_parts += [r]
        state_parts += [z]
        final_state = torch.cat(state_parts, dim=-1)
        return y_hat, final_state



def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return FSMRecurrentModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars.get("n_hidden_states", 64), hidden_sizes=hidden_sizes,
            num_layers=config_pars.get("num_layers", 1), recurrent=model_type, activation=activation,
            dropout_prob=dropout_prob, direct_feedthrough=config_pars.get("direct_feedthrough", False),
        )

    if model_type == "HYSTERESIS_LFR":
        return FSMHysteresisLFRModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_complex_modes=config_pars.get("n_complex_modes", 3), n_real_modes=config_pars.get("n_real_modes", 2),
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
            sampling_time=config_pars.get("sampling_time", 1.0 / 6400.0), e_pos=config_pars.get("e_pos", 0.05),
            pole_freq_hz_init=config_pars.get("pole_freq_hz_init", None),
            pole_damping_ratio_init=config_pars.get("pole_damping_ratio_init", 0.02),
            real_decay_hz_init=config_pars.get("real_decay_hz_init", None),
        )

    raise ValueError("Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'HYSTERESIS_LFR'}.")