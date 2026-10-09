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


class F16RecurrentModel(nn.Module):
    """Black-box recurrent baseline (matches the leaderboard's own
    RNN/GRU/LSTM entries). A small MLP maps the initial-condition vector
    y0 (built from the last `history_window` force+acceleration samples)
    to the initial hidden state of an RNN/GRU/LSTM that then rolls
    forward over the force input, producing all 3 acceleration channels
    jointly."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=3, n_hidden_states=64, hidden_sizes=None,
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


class F16ModalWhiteBoxModel(nn.Module):
    """
    Genuine white-box model, directly reflecting the benchmark's own
    documented physics (F16Benchmark.pdf, Section 5 -- read this session):

    "the mounting interface... [features] nonlinearities in STIFFNESS and
    DAMPING, due to CLEARANCE and FRICTION respectively" -- and "hard
    nonlinearities... may not be appropriately modeled using smooth
    basis functions." The paper's own suggested simplification: "focus
    on [the] latter [wing torsion] mode" (~7.3Hz) rather than the full
    ~10-mode, 2-15Hz system.

    Structure: `n_modes` independent linear modal oscillators (default 2:
    the ~5.2Hz wing-bending mode and the ~7.3Hz wing-torsion mode, the
    two modes the paper specifically names), each driven by the input
    force through its own learnable participation factor, PLUS one
    shared nonlinear restoring force acting at the mounting interface --
    modeled as a learnable linear combination of the modal coordinates
    (`z = sum_i(c_i * q_i)`, representing the interface's relative
    motion), fed back into every mode weighted by that SAME participation
    factor (standard modal-projection consistency: a physical force at
    one location projects onto each mode in proportion to that mode's
    own shape at that location):

        q_i'' + 2*zeta_i*omega_i*q_i' + omega_i^2*q_i
            = c_i*(b_gain*u(t) - f_clearance(z) - f_friction(zdot))
        z = sum_i(c_i * q_i)

    The 3 measured accelerations are each a learnable linear combination
    of the modal accelerations (`y_j = sum_i(d_{j,i} * q_i'')`) --
    standard modal superposition: every sensor sees a mix of every
    excited mode, weighted by that mode's shape at the sensor's specific
    location.

    Nonlinearity, matching the paper's own description directly:
      - `f_clearance(z)`: a smooth relaxation of a piecewise/dead-zone
        spring (`softplus`-based, not `relu`, so it stays differentiable
        everywhere) -- zero force until |z| exceeds a learnable
        `clearance_gap`, then engages with stiffness `k_nl`. This is
        exactly the mechanism the paper attributes to the T-shaped
        connector sliding through its rail: no force until the gap
        closes.
      - `f_friction(zdot)`: a smooth relaxation of Coulomb friction
        (`tanh(zdot/eps_vel)`, matching the same smoothing convention
        already used for EMPS's own `Fc*sign(qd)` term elsewhere in this
        collection) -- roughly constant-magnitude force opposing
        relative velocity at the interface, independent of its size.

    Integrated with a FIXED (not learned) discrete-time recursion,
    `dt = Ts/num_substeps`, Ts = 1/fs -- matching the established, proven
    convention from every other physics-integrated model in this
    collection (learned/dt-coupled integration steps have repeatedly
    caused gradient-starvation bugs elsewhere; fixed dt avoids that
    class of problem from the start).
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=3, n_modes=2, hidden_sizes=None,
                 activation="Tanh", dropout_prob=0.0, num_substeps=4, sampling_time=1.0 / 400.0,
                 omega_init_hz=None, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if omega_init_hz is None:
            # Defaults: paper's own named frequencies -- wing bending
            # (~5.2Hz) and wing torsion (~7.3Hz, "the mode involving the
            # most substantial nonlinear distortions"). Extra modes (if
            # n_modes > 2) get spread across the excited 2-15Hz band.
            base = [5.2, 7.3]
            omega_init_hz = base[:n_modes] if n_modes <= 2 else base + list(
                torch.linspace(8.0, 14.0, n_modes - 2).tolist()
            )
        if len(omega_init_hz) != n_modes:
            raise ValueError("omega_init_hz must have exactly n_modes entries.")

        self.n_modes = n_modes
        self.num_substeps = int(num_substeps)
        self.dt = float(sampling_time) / self.num_substeps

        self.state_init = FeedforwardNetwork(n_states, hidden_sizes, 2 * n_modes, activation=activation, dropout_prob=dropout_prob)
        nn.init.zeros_(self.state_init.net[-1].weight)
        nn.init.zeros_(self.state_init.net[-1].bias)

        # Modal frequencies (rad/s) -- kept strictly positive via softplus
        # over a raw parameter, initialized at the paper's own named
        # values (converted from Hz).
        omega_init_rad = torch.tensor([2 * 3.141592653589793 * f for f in omega_init_hz], dtype=torch.float32)
        self.omega_raw = nn.Parameter(torch.log(torch.expm1(omega_init_rad)))  # softplus(raw) = omega_init_rad exactly

        # Damping ratios -- kept in (0, 1) via sigmoid, light structural
        # damping (~1%) is a reasonable generic starting point for an
        # aircraft structure's linear modes.
        zeta_init = 0.01
        zeta_raw_init = torch.log(torch.tensor(zeta_init / (1 - zeta_init)))
        self.zeta_raw = nn.Parameter(torch.full((n_modes,), float(zeta_raw_init)))

        # Modal force-participation / interface-coupling factors -- same
        # learnable c_i used both for how strongly u(t) drives mode i and
        # how strongly the shared nonlinear interface force feeds back
        # into mode i (modal-projection consistency, see class docstring).
        self.c_participation = nn.Parameter(torch.ones(n_modes) / n_modes)
        self.b_gain = nn.Parameter(torch.tensor(1.0))

        # Sensor mode-shape matrix: y_j = sum_i(d_{j,i} * qddot_i).
        self.d_output = nn.Parameter(torch.randn(n_outputs, n_modes) * 0.1 + 1.0 / n_modes)

        # Clearance (nonlinear stiffness) parameters.
        self.k_nl_raw = nn.Parameter(torch.tensor(0.0))  # softplus(0)=0.69, modest starting stiffness
        self.clearance_gap_raw = nn.Parameter(torch.tensor(0.0))  # softplus(0)=0.69, modest starting gap
        # Friction (nonlinear damping) parameters.
        self.c_nl_raw = nn.Parameter(torch.tensor(0.0))
        self.friction_vel_eps = 0.01  # fixed smoothing width for the tanh relaxation, not learned

    def _modal_matrices(self):
        omega = nn.functional.softplus(self.omega_raw)  # (n_modes,), always > 0
        zeta = torch.sigmoid(self.zeta_raw)  # (n_modes,), in (0,1)
        return omega, zeta

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        state0 = self.state_init(y0)  # (batch, 2*n_modes): [q_1..q_n, qdot_1..qdot_n]
        q = state0[:, : self.n_modes]
        qdot = state0[:, self.n_modes:]

        omega, zeta = self._modal_matrices()
        k_nl = nn.functional.softplus(self.k_nl_raw)
        clearance_gap = nn.functional.softplus(self.clearance_gap_raw)
        c_nl = nn.functional.softplus(self.c_nl_raw)

        outputs = []
        for t in range(T):
            u_t = u[:, t, :].squeeze(-1)  # (batch,)
            for _ in range(self.num_substeps):
                z = torch.sum(self.c_participation.unsqueeze(0) * q, dim=-1)  # (batch,) interface displacement
                zdot = torch.sum(self.c_participation.unsqueeze(0) * qdot, dim=-1)  # (batch,) interface velocity

                # Smooth dead-zone (clearance) spring: zero force inside
                # the gap, engages smoothly beyond it, either direction.
                f_clearance = k_nl * (nn.functional.softplus(z - clearance_gap) - nn.functional.softplus(-z - clearance_gap))
                # Smooth Coulomb friction: roughly constant-magnitude,
                # opposing the sign of the interface velocity.
                f_friction = c_nl * torch.tanh(zdot / self.friction_vel_eps)
                f_interface = f_clearance + f_friction  # (batch,)

                qddot = -2 * zeta.unsqueeze(0) * omega.unsqueeze(0) * qdot - (omega.unsqueeze(0) ** 2) * q \
                        + self.c_participation.unsqueeze(0) * (self.b_gain * u_t.unsqueeze(-1) - f_interface.unsqueeze(-1))

                qdot = qdot + self.dt * qddot
                q = q + self.dt * qdot

            y_t = torch.matmul(qddot, self.d_output.t())  # (batch, n_outputs) -- modal superposition of accelerations
            outputs.append(y_t.unsqueeze(1))

        y_hat = torch.cat(outputs, dim=1)
        final_state = torch.cat([q, qdot], dim=-1)
        return y_hat, final_state


class F16ModalConvModel(nn.Module):
    """Same 2-mode linear-modal physics as `F16ModalWhiteBoxModel` (paper's
    wing-bending/wing-torsion modes, Section 5), but solved in closed form
    instead of a per-timestep Python-loop ODE integrator: the per-mode
    free response (from the y0-derived initial state) has an analytic
    damped-sinusoid formula, and the forced response is the convolution of
    the input force with each mode's analytic impulse response, evaluated
    via FFT (`torch.fft.rfft`/`irfft`) over a `kernel_length`-sample
    truncated kernel (valid because a lightly-damped mode's impulse
    response decays to ~0 well within a few thousand samples). This is
    fully vectorized over time -- no Python loop over T -- which matters
    because F16 training/validation sequences run 70k-117k samples long;
    the ODE-loop version (`MODAL_WHITEBOX`) could not get through even one
    fold's time budget on this benchmark. Nonlinear friction/clearance are
    dropped here (that's what decouples the modes and makes the per-mode
    convolution valid); a residual correction for those effects belongs in
    a HYBRID model built on top of this backbone.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=3, n_modes=2, hidden_sizes=None,
                 activation="Tanh", dropout_prob=0.0, kernel_length=8192, sampling_time=1.0 / 400.0,
                 omega_init_hz=None, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1")
        if omega_init_hz is None:
            base = [5.2, 7.3]
            omega_init_hz = base[:n_modes] if n_modes <= 2 else base + list(
                torch.linspace(8.0, 14.0, n_modes - 2).tolist()
            )
        if len(omega_init_hz) != n_modes:
            raise ValueError("omega_init_hz must have exactly n_modes entries.")

        self.n_modes = n_modes
        self.kernel_length = int(kernel_length)
        self.dt = float(sampling_time)

        self.state_init = FeedforwardNetwork(n_states, hidden_sizes, 2 * n_modes, activation=activation, dropout_prob=dropout_prob)
        nn.init.zeros_(self.state_init.net[-1].weight)
        nn.init.zeros_(self.state_init.net[-1].bias)

        omega_init_rad = torch.tensor([2 * 3.141592653589793 * f for f in omega_init_hz], dtype=torch.float32)
        self.omega_raw = nn.Parameter(torch.log(torch.expm1(omega_init_rad)))
        zeta_init = 0.01
        zeta_raw_init = torch.log(torch.tensor(zeta_init / (1 - zeta_init)))
        self.zeta_raw = nn.Parameter(torch.full((n_modes,), float(zeta_raw_init)))

        self.c_participation = nn.Parameter(torch.ones(n_modes) / n_modes)
        self.b_gain = nn.Parameter(torch.tensor(1.0))
        self.d_output = nn.Parameter(torch.randn(n_outputs, n_modes) * 0.1 + 1.0 / n_modes)

    def _modal_matrices(self):
        omega = nn.functional.softplus(self.omega_raw)
        zeta = torch.sigmoid(self.zeta_raw)
        omega_d = omega * torch.sqrt(torch.clamp(1 - zeta ** 2, min=1e-4))
        return omega, zeta, omega_d

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        batch_size, T, _ = u.shape
        device, dtype = u.device, u.dtype

        state0 = self.state_init(y0)
        q0 = state0[:, : self.n_modes]
        qdot0 = state0[:, self.n_modes:]
        omega, zeta, omega_d = self._modal_matrices()

        t = torch.arange(T, device=device, dtype=dtype) * self.dt
        decay_t = torch.exp(-zeta.unsqueeze(-1) * omega.unsqueeze(-1) * t.unsqueeze(0))  # (n_modes, T)
        cos_t = torch.cos(omega_d.unsqueeze(-1) * t.unsqueeze(0))
        sin_t = torch.sin(omega_d.unsqueeze(-1) * t.unsqueeze(0))

        q0e = q0.unsqueeze(-1)  # (batch, n_modes, 1)
        qdot0e = qdot0.unsqueeze(-1)
        omega_e = omega.view(1, -1, 1)
        zeta_e = zeta.view(1, -1, 1)
        omega_d_e = omega_d.view(1, -1, 1)

        sin_coef = (qdot0e + zeta_e * omega_e * q0e) / omega_d_e
        q_free = decay_t.unsqueeze(0) * (q0e * cos_t.unsqueeze(0) + sin_coef * sin_t.unsqueeze(0))
        qdot_sin_coef = ((omega_e ** 2) * q0e + zeta_e * omega_e * qdot0e) / omega_d_e
        qdot_free = decay_t.unsqueeze(0) * (qdot0e * cos_t.unsqueeze(0) - qdot_sin_coef * sin_t.unsqueeze(0))

        L = min(self.kernel_length, T)
        tau = torch.arange(L, device=device, dtype=dtype) * self.dt
        decay_k = torch.exp(-zeta.unsqueeze(-1) * omega.unsqueeze(-1) * tau.unsqueeze(0))  # (n_modes, L)
        sin_k = torch.sin(omega_d.unsqueeze(-1) * tau.unsqueeze(0))
        cos_k = torch.cos(omega_d.unsqueeze(-1) * tau.unsqueeze(0))
        h = decay_k * sin_k / omega_d.unsqueeze(-1)
        hdot = decay_k * (cos_k - (zeta.unsqueeze(-1) * omega.unsqueeze(-1) / omega_d.unsqueeze(-1)) * sin_k)

        u_t = u.squeeze(-1)  # (batch, T)
        f = self.c_participation.view(1, -1, 1) * self.b_gain * u_t.unsqueeze(1)  # (batch, n_modes, T)

        fft_len = 1
        while fft_len < T + L - 1:
            fft_len *= 2
        F_f = torch.fft.rfft(f, n=fft_len, dim=-1)
        H_h = torch.fft.rfft(h, n=fft_len, dim=-1).unsqueeze(0)
        H_hdot = torch.fft.rfft(hdot, n=fft_len, dim=-1).unsqueeze(0)
        conv_q = torch.fft.irfft(F_f * H_h, n=fft_len, dim=-1)[..., :T] * self.dt
        conv_qdot = torch.fft.irfft(F_f * H_hdot, n=fft_len, dim=-1)[..., :T] * self.dt

        q = q_free + conv_q  # (batch, n_modes, T)
        qdot = qdot_free + conv_qdot
        qddot = -2 * zeta_e * omega_e * qdot - (omega_e ** 2) * q + self.c_participation.view(1, -1, 1) * self.b_gain * u_t.unsqueeze(1)

        y_hat = torch.einsum("bnt,on->bto", qddot, self.d_output)
        final_state = torch.cat([q[..., -1], qdot[..., -1]], dim=-1)
        return y_hat, final_state


class F16HybridModel(nn.Module):
    """Grey-box: `F16ModalConvModel` physical backbone (2-mode linear
    modal physics, fast/vectorized) plus a small `LSTM` residual reading
    `[u(t), y_phys(t)]` that corrects for what the linear backbone can't
    represent -- the paper's documented nonlinear clearance/friction at
    the mounting interface, and modes outside the named 5.2/7.3Hz pair.
    `nn.LSTM` specifically (not GRU/RNN) because this environment's
    PyTorch CPU build makes LSTM the only recurrent cell fast enough to
    run on F16's 70k-117k-sample sequences within the fold time budget.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=3, n_modes=2, hidden_sizes=None,
                 activation="Tanh", dropout_prob=0.0, kernel_length=8192, sampling_time=1.0 / 400.0,
                 omega_init_hz=None, residual_hidden_size=32, residual_num_layers=1, **kwargs):
        super().__init__()
        self.backbone = F16ModalConvModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs, n_modes=n_modes,
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
            kernel_length=kernel_length, sampling_time=sampling_time, omega_init_hz=omega_init_hz,
        )
        self.residual_hidden_size = int(residual_hidden_size)
        self.residual_lstm = nn.LSTM(
            input_size=n_inputs + n_outputs, hidden_size=self.residual_hidden_size,
            num_layers=residual_num_layers, batch_first=True,
        )
        self.residual_out = nn.Linear(self.residual_hidden_size, n_outputs)
        # Start with a near-zero residual so early training relies on the
        # (already-physically-sensible) backbone rather than a randomly
        # initialized correction fighting it from step 0.
        nn.init.zeros_(self.residual_out.weight)
        nn.init.zeros_(self.residual_out.bias)

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_phys, backbone_state = self.backbone(u, y0)
        residual_input = torch.cat([u, y_phys], dim=-1)
        lstm_out, _ = self.residual_lstm(residual_input)
        residual = self.residual_out(lstm_out)
        y_hat = y_phys + residual
        return y_hat, backbone_state


class CausalConvBlock(nn.Module):
    """One dilated causal conv1d residual block (WaveNet-style), with the
    y0-conditioning vector injected as an additive bias before the
    activation (FiLM-style shift only, no scale -- simplest form that
    still lets the initial-condition context modulate every layer)."""

    def __init__(self, channels, kernel_size, dilation, cond_dim, activation):
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.cond_proj = nn.Linear(cond_dim, channels)
        self.out_proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.act = make_activation(activation)

    def forward(self, x, cond):
        pad = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (pad, 0))
        h = self.conv(x_padded) + self.cond_proj(cond).unsqueeze(-1)
        h = self.act(h)
        h = self.out_proj(h)
        return x + h


class F16CausalCNNModel(nn.Module):
    """Black-box alternative to the recurrent baselines: a stack of
    dilated causal conv1d residual blocks (WaveNet-style), fully
    vectorized over time (no sequential recurrence at all, unlike
    RNN/GRU/LSTM), conditioned on the y0-derived initial-condition vector
    via an additive bias injected into every block. Dilations double each
    block (1,2,4,...) so `n_layers` blocks give a receptive field of
    `1 + (kernel_size-1)*(2**n_layers - 1)` samples of true causal
    context, without unrolling a Python loop over T."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=3, channels=32, n_layers=7, kernel_size=3,
                 hidden_sizes=None, activation="ReLU", dropout_prob=0.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32]
        cond_dim = channels
        self.state_init = FeedforwardNetwork(n_states, hidden_sizes, cond_dim, activation, dropout_prob)
        self.input_proj = nn.Conv1d(n_inputs, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalConvBlock(channels, kernel_size, dilation=2 ** i, cond_dim=cond_dim, activation=activation)
            for i in range(n_layers)
        ])
        self.output_proj = nn.Conv1d(channels, n_outputs, kernel_size=1)

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        cond = self.state_init(y0)  # (batch, cond_dim)
        x = self.input_proj(u.transpose(1, 2))  # (batch, channels, T)
        for block in self.blocks:
            x = block(x, cond)
        y_hat = self.output_proj(x).transpose(1, 2)  # (batch, T, n_outputs)
        return y_hat, cond


class F16HybridTCNModel(nn.Module):
    """Grey-box variant of `F16HybridModel`: same `MODAL_CONV` physical
    backbone, but the residual correction is a dilated causal conv1d
    stack (`CausalConvBlock`, same as `F16CausalCNNModel`) reading
    `[u(t), y_phys(t)]` instead of an LSTM -- since the standalone TCN
    (run08/run09) clearly outperformed the standalone LSTM on this
    benchmark, this tests whether a TCN-based residual on top of the
    physics backbone beats the best pure-black-box TCN outright."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=3, n_modes=4, hidden_sizes=None,
                 activation="Tanh", dropout_prob=0.0, kernel_length=8192, sampling_time=1.0 / 400.0,
                 omega_init_hz=None, channels=64, n_layers=9, kernel_size=3, **kwargs):
        super().__init__()
        self.backbone = F16ModalConvModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs, n_modes=n_modes,
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
            kernel_length=kernel_length, sampling_time=sampling_time, omega_init_hz=omega_init_hz,
        )
        self.state_init = FeedforwardNetwork(n_states, hidden_sizes or [32], channels, activation, dropout_prob)
        self.input_proj = nn.Conv1d(n_inputs + n_outputs, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            CausalConvBlock(channels, kernel_size, dilation=2 ** i, cond_dim=channels, activation=activation)
            for i in range(n_layers)
        ])
        self.output_proj = nn.Conv1d(channels, n_outputs, kernel_size=1)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_phys, backbone_state = self.backbone(u, y0)
        cond_y0 = y0.squeeze(1) if y0.dim() == 3 else y0
        cond = self.state_init(cond_y0)
        x = self.input_proj(torch.cat([u, y_phys], dim=-1).transpose(1, 2))
        for block in self.blocks:
            x = block(x, cond)
        residual = self.output_proj(x).transpose(1, 2)
        y_hat = y_phys + residual
        return y_hat, backbone_state


def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return F16RecurrentModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars.get("n_hidden_states", 64), hidden_sizes=hidden_sizes,
            num_layers=config_pars.get("num_layers", 1), recurrent=model_type, activation=activation,
            dropout_prob=dropout_prob, direct_feedthrough=config_pars.get("direct_feedthrough", False),
        )

    if model_type == "MODAL_WHITEBOX":
        return F16ModalWhiteBoxModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_modes=config_pars.get("n_modes", 2), hidden_sizes=hidden_sizes, activation=activation,
            dropout_prob=dropout_prob, num_substeps=config_pars.get("num_substeps", 4),
            sampling_time=config_pars.get("sampling_time", 1.0 / 400.0),
            omega_init_hz=config_pars.get("omega_init_hz", None),
        )

    if model_type == "MODAL_CONV":
        return F16ModalConvModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_modes=config_pars.get("n_modes", 2), hidden_sizes=hidden_sizes, activation=activation,
            dropout_prob=dropout_prob, kernel_length=config_pars.get("kernel_length", 8192),
            sampling_time=config_pars.get("sampling_time", 1.0 / 400.0),
            omega_init_hz=config_pars.get("omega_init_hz", None),
        )

    if model_type == "HYBRID":
        return F16HybridModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_modes=config_pars.get("n_modes", 2), hidden_sizes=hidden_sizes, activation=activation,
            dropout_prob=dropout_prob, kernel_length=config_pars.get("kernel_length", 8192),
            sampling_time=config_pars.get("sampling_time", 1.0 / 400.0),
            omega_init_hz=config_pars.get("omega_init_hz", None),
            residual_hidden_size=config_pars.get("residual_hidden_size", 32),
            residual_num_layers=config_pars.get("residual_num_layers", 1),
        )

    if model_type == "TCN":
        return F16CausalCNNModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            channels=config_pars.get("channels", 32), n_layers=config_pars.get("n_layers", 7),
            kernel_size=config_pars.get("kernel_size", 3), hidden_sizes=hidden_sizes,
            activation=activation, dropout_prob=dropout_prob,
        )

    if model_type == "HYBRID_TCN":
        return F16HybridTCNModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_modes=config_pars.get("n_modes", 4), hidden_sizes=hidden_sizes, activation=activation,
            dropout_prob=dropout_prob, kernel_length=config_pars.get("kernel_length", 8192),
            sampling_time=config_pars.get("sampling_time", 1.0 / 400.0),
            omega_init_hz=config_pars.get("omega_init_hz", None),
            channels=config_pars.get("channels", 64), n_layers=config_pars.get("n_layers", 9),
            kernel_size=config_pars.get("kernel_size", 3),
        )

    raise ValueError(
        "Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'MODAL_WHITEBOX', 'MODAL_CONV', "
        "'HYBRID', 'TCN', 'HYBRID_TCN'}."
    )