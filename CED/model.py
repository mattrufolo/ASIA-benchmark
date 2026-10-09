from __future__ import annotations

import warnings

import numpy as np
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
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
    ) -> None:
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
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        recurrent: str = "LSTM",
        dropout_prob: float = 0.0,
    ) -> None:
        super().__init__()

        recurrent_key = recurrent.upper()
        common_args = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": True,
            "dropout": dropout_prob if num_layers > 1 else 0.0,
        }

        if recurrent_key == "LSTM":
            self.recurrent = nn.LSTM(**common_args)
        elif recurrent_key == "GRU":
            self.recurrent = nn.GRU(**common_args)
        elif recurrent_key == "RNN":
            self.recurrent = nn.RNN(**common_args)
        else:
            raise ValueError("recurrent must be one of {'RNN', 'GRU', 'LSTM'}")

        self.recurrent_type = recurrent_key
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        output, hidden_state = self.recurrent(x, hidden_state)
        y = self.output_layer(output)
        return y, hidden_state


class CEDRecurrentModel(nn.Module):
    """Generic black-box recurrent model.

    A small MLP maps the initial-condition vector y0 (built from the last
    `history_window` input/output pairs) to the initial hidden state of an
    RNN/GRU/LSTM that then rolls forward over the input trajectory u.
    """

    def __init__(
        self,
        n_inputs: int = 1,
        n_states: int = 1,
        n_outputs: int = 1,
        n_hidden_states: int = 32,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 1,
        recurrent: str = "LSTM",
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
        direct_feedthrough: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32]

        self.n_inputs = n_inputs
        self.n_states = n_states
        self.n_outputs = n_outputs
        self.n_hidden_states = n_hidden_states
        self.num_layers = num_layers
        self.recurrent_type = recurrent.upper()
        self.direct_feedthrough = direct_feedthrough

        self.initial_state_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.recurrent_block = RecurrentBlock(
            input_size=n_inputs,
            hidden_size=n_hidden_states,
            output_size=n_outputs,
            num_layers=num_layers,
            recurrent=recurrent,
            dropout_prob=dropout_prob,
        )

        self.direct_layer = None
        if direct_feedthrough:
            self.direct_layer = nn.Linear(n_inputs, n_outputs)

    def build_initial_state(
        self, y0: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        if y0.dim() != 2:
            raise ValueError("y0 must have shape (batch, n_states) or (batch, 1, n_states)")

        hidden_single = self.initial_state_net(y0)
        hidden = hidden_single.unsqueeze(0).repeat(self.num_layers, 1, 1)

        if self.recurrent_type == "LSTM":
            cell = hidden.clone()
            return hidden, cell
        return hidden

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, object]:
        initial_state = self.build_initial_state(y0)
        y_hat, hidden_state = self.recurrent_block(u, initial_state)
        if self.direct_layer is not None:
            y_hat = y_hat + self.direct_layer(u)
        return y_hat, hidden_state


class CausalConv1d(nn.Conv1d):
    """1D convolution padded on the left only, so output[t] never depends on
    input[t'] for t' > t (strict causality, required since this model rolls
    forward over a trajectory the same way the recurrent models do)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.left_padding, 0))
        return super().forward(x)


class TCNResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, activation: str, dropout_prob: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.act1 = make_activation(activation)
        self.act2 = make_activation(activation)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act1(self.conv1(x))
        out = self.dropout(out)
        out = self.act2(self.conv2(out))
        return x + out


class CEDTCNModel(nn.Module):
    """Causal dilated-convolutional (TCN) black-box model.

    Unlike the recurrent models, a TCN has no explicit carried hidden state
    to initialize from `y0` -- so instead `y0` is mapped (via a small MLP) to
    a fixed context vector that is broadcast and concatenated to `u` at every
    timestep before the first conv layer. This gives every block a summary of
    the pre-window history without needing raw padded samples, similar in
    spirit to how the recurrent models' initial hidden state is built.
    Dilations double each block (1, 2, 4, ...), giving an exponentially
    growing receptive field at linear parameter cost -- a structurally
    different way of capturing the lightly-damped resonance than a
    fixed-size recurrent state.
    """

    def __init__(
        self,
        n_inputs: int = 1,
        n_states: int = 1,
        n_outputs: int = 1,
        tcn_channels: int = 32,
        kernel_size: int = 3,
        num_blocks: int = 5,
        activation: str = "ReLU",
        dropout_prob: float = 0.0,
        context_hidden_sizes: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if context_hidden_sizes is None:
            context_hidden_sizes = [32]

        self.context_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=context_hidden_sizes,
            output_size=tcn_channels,
            activation=activation,
            dropout_prob=dropout_prob,
        )
        self.input_proj = nn.Conv1d(n_inputs + tcn_channels, tcn_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    channels=tcn_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    activation=activation,
                    dropout_prob=dropout_prob,
                )
                for block_index in range(num_blocks)
            ]
        )
        self.output_layer = nn.Conv1d(tcn_channels, n_outputs, kernel_size=1)

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, seq_len, _ = u.shape
        context = self.context_net(y0)  # (batch, tcn_channels)
        context_expanded = context.unsqueeze(-1).expand(-1, -1, seq_len)

        x = u.transpose(1, 2)  # (batch, n_inputs, T)
        x = torch.cat([x, context_expanded], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        y_hat = self.output_layer(x).transpose(1, 2)  # (batch, T, n_outputs)
        return y_hat, x


def init_companion_coeffs(
    n_ode_states: int,
    pole_init: float,
    resonant_init: bool,
    alpha_init: float,
    xi_init: float,
    omega0_init: float,
    sampling_time: float,
) -> torch.Tensor:
    """Shared companion-form coefficient initializer used by every
    physical/hybrid model here (see WienerCEDModel's docstring for the full
    rationale). Extracted so PhysicsGuidedTCNModel can reuse the exact same,
    already-verified pole-init logic instead of duplicating it."""
    if n_ode_states < 1:
        raise ValueError("n_ode_states must be >= 1")
    if resonant_init and n_ode_states != 3:
        raise ValueError("resonant_init requires n_ode_states=3 (one real pole + one complex pair).")
    if not resonant_init and not (-1.0 < pole_init < 1.0):
        raise ValueError(
            "pole_init is a DISCRETE-time pole and must lie strictly inside the unit "
            "circle (-1, 1) for the initial recursion to be stable."
        )

    if resonant_init:
        real_pole_c = -float(alpha_init)
        xi_v, omega0_v = float(xi_init), float(omega0_init)
        complex_pole_c = complex(-xi_v * omega0_v, omega0_v * (1.0 - xi_v**2) ** 0.5)
        ts = float(sampling_time)
        discrete_poles = np.array(
            [
                np.exp(real_pole_c * ts),
                np.exp(complex_pole_c * ts),
                np.exp(np.conj(complex_pole_c) * ts),
            ]
        )
        if np.any(np.abs(discrete_poles) >= 1.0):
            raise ValueError("resonant_init poles are not strictly inside the unit circle; check alpha/xi/omega0_init.")
        char_poly = np.real(np.poly(discrete_poles))  # [1, c1, c2, c3], z^3 + c1 z^2 + c2 z + c3
        a_init_np = np.array([-char_poly[3], -char_poly[2], -char_poly[1]])
        return torch.tensor(a_init_np, dtype=torch.float32)

    from math import comb

    p = float(pole_init)
    a_init = torch.zeros(n_ode_states)
    for k in range(n_ode_states):
        a_init[k] = ((-1) ** (n_ode_states - k + 1)) * comb(n_ode_states, k) * (p ** (n_ode_states - k))
    return a_init


class PhysicsGuidedTCNModel(nn.Module):
    """Tighter physics/black-box coupling than HybridTCNCEDModel: instead of
    summing an independent TCN residual on top of the physical model's final
    rectified output, this rolls the SAME companion-form Wiener ODE forward
    and exposes its full per-timestep internal state trajectory (pre-
    rectifier, `n_ode_states` channels) as extra input channels to the TCN --
    alongside `u` and the usual y0-context channel. The TCN can therefore
    learn corrections that depend on the physical model's own internal state
    (e.g. near the rectifier's non-differentiable point at x1=0), not just
    on the raw input trajectory. The physical rectified output and the TCN
    output are still combined additively (`y_hat = y_phys + scale * y_tcn`),
    keeping the same stable-by-construction physical skip connection as
    HybridTCNCEDModel.
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_ode_states: int = 3,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        pole_init: float = 0.7,
        resonant_init: bool = False,
        alpha_init: float = 2.0,
        xi_init: float = 0.1,
        omega0_init: float = 25.0,
        sampling_time: float = 0.02,
        tcn_channels: int = 32,
        kernel_size: int = 3,
        num_blocks: int = 5,
        context_hidden_sizes: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if context_hidden_sizes is None:
            context_hidden_sizes = hidden_sizes

        self.n_ode_states = n_ode_states

        self.state_init = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_ode_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )
        self.b_gain = nn.Parameter(torch.tensor(0.1))
        self.a_coeffs = nn.Parameter(
            init_companion_coeffs(
                n_ode_states=n_ode_states,
                pole_init=pole_init,
                resonant_init=resonant_init,
                alpha_init=alpha_init,
                xi_init=xi_init,
                omega0_init=omega0_init,
                sampling_time=sampling_time,
            )
        )
        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

        self.context_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=context_hidden_sizes,
            output_size=tcn_channels,
            activation=activation,
            dropout_prob=dropout_prob,
        )
        self.input_proj = nn.Conv1d(n_inputs + n_ode_states + tcn_channels, tcn_channels, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TCNResidualBlock(
                    channels=tcn_channels,
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    activation=activation,
                    dropout_prob=dropout_prob,
                )
                for block_index in range(num_blocks)
            ]
        )
        self.output_layer = nn.Conv1d(tcn_channels, n_outputs, kernel_size=1)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, seq_len, _ = u.shape
        x = self.state_init(y0)  # (batch, n_ode_states)

        phys_outputs = []
        state_trajectory = []
        for t in range(seq_len):
            u_t = u[:, t, :].squeeze(-1)
            if self.n_ode_states == 1:
                x = (self.b_gain * u_t + self.a_coeffs[0] * x[:, 0]).unsqueeze(-1)
            else:
                x_shift = x[:, 1:]
                x_last = self.b_gain * u_t + torch.sum(self.a_coeffs.unsqueeze(0) * x, dim=-1)
                x = torch.cat([x_shift, x_last.unsqueeze(-1)], dim=-1)

            x1 = x[:, 0:1]
            phys_outputs.append(self.out_scale * torch.abs(x1) + self.out_bias)
            state_trajectory.append(x)

        y_phys = torch.stack(phys_outputs, dim=1)  # (batch, T, 1)
        x_traj = torch.stack(state_trajectory, dim=1)  # (batch, T, n_ode_states)

        context = self.context_net(y0)  # (batch, tcn_channels)
        context_expanded = context.unsqueeze(-1).expand(-1, -1, seq_len)

        u_ch = u.transpose(1, 2)  # (batch, n_inputs, T)
        x_ch = x_traj.transpose(1, 2)  # (batch, n_ode_states, T)
        combined = torch.cat([u_ch, x_ch, context_expanded], dim=1)

        h = self.input_proj(combined)
        for block in self.blocks:
            h = block(h)
        y_res = self.output_layer(h).transpose(1, 2)  # (batch, T, n_outputs)

        y_hat = y_phys + self.residual_scale * y_res
        return y_hat, x_traj


class WienerCEDModel(nn.Module):
    """
    Grey-box Wiener model inspired by eqs. (1)-(2) / (6)-(8) of the CED
    technical report: a linear companion-form recursion of order
    `n_ode_states` followed by a static output rectifier (the pulse-counter
    speed sensor is insensitive to the sign of the velocity).

    Discrete-time companion recursion (one step per real sample, matching
    the report's own eq. 9 discretization rather than a continuous-time ODE):
        x_1[k+1] = x_2[k]
        x_2[k+1] = x_3[k]
        ...
        x_n[k+1] = b*u[k] + a_1*x_n[k] + a_2*x_{n-1}[k] + ... + a_n*x_1[k]
        y[k]     = out_scale * |x_1[k]| + out_bias

    NOTE on an earlier, discarded version of this model: a previous
    implementation integrated a continuous-time ODE with forward Euler and a
    learnable step size `dt` (dx = ...; x <- x + dt*dx). That multiplies
    EVERY gradient reaching `a_coeffs`/`b_gain` by `dt`, and the gradient
    reaching `log_dt` itself by another factor of `dt` (since
    d(exp(log_dt))/d(log_dt) = dt). With a physically-realistic dt=0.02,
    this made the dynamics parameters' gradients ~100-1000x smaller than
    `out_scale`/`out_bias`/`state_init`'s gradients, so optimization only
    ever moved the cheap output-affine terms and never learned real
    dynamics -- exactly the "loss frozen near its untrained value,
    best_epoch=0-1" failure observed in practice. Removing `dt` entirely
    (direct discrete recursion) fixes this: verified that `a_coeffs`/`b_gain`
    then receive the LARGEST gradients in the model, as they should, since
    they control the dynamics.

    The rectifier also uses true `torch.abs` rather than a smooth surrogate
    like `sqrt(x^2 + eps)`: the latter's gradient vanishes near x=0, which
    this benchmark hits constantly (the belt reverses direction often).
    `torch.abs` has gradient magnitude exactly 1 everywhere except the
    single point x=0 (subgradient 0 there), so there is no shrinking
    near-zero dead zone.

    Pole initialization: by default (`resonant_init=False`) all n_ode_states
    poles start as one repeated real pole at `pole_init`, which can only
    produce monotonic decay. `CED_description.md` Section 2 states the
    physical system is actually
        (s + alpha) * (s^2 + 2*xi*omega_0*s + omega_0^2)
    i.e. one real pole (the drives) times a lightly-damped complex-conjugate
    pair (the spring/belt resonance, reported peak ~25 rad/s). Setting
    `resonant_init=True` (requires n_ode_states=3) instead builds the initial
    companion coefficients from that structure: the continuous-time poles
    `-alpha_init` and `-xi_init*omega0_init +/- j*omega0_init*sqrt(1-xi^2)`
    are mapped to discrete-time poles via the matched pole method
    (`p_d = exp(p_c * Ts)`, which exactly preserves damping ratio and natural
    frequency under sampling), and the discrete companion coefficients are
    read off `numpy.poly` of those three discrete poles. This starts gradient
    descent inside the right qualitative regime (oscillatory + slow decay)
    instead of a purely-decaying one.
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_ode_states: int = 3,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        pole_init: float = 0.7,
        resonant_init: bool = False,
        alpha_init: float = 2.0,
        xi_init: float = 0.1,
        omega0_init: float = 25.0,
        sampling_time: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if n_ode_states < 1:
            raise ValueError("n_ode_states must be >= 1")
        if resonant_init and n_ode_states != 3:
            raise ValueError("resonant_init requires n_ode_states=3 (one real pole + one complex pair).")
        if not resonant_init and not (-1.0 < pole_init < 1.0):
            raise ValueError(
                "pole_init is a DISCRETE-time pole and must lie strictly inside the unit "
                "circle (-1, 1) for the initial recursion to be stable."
            )

        self.n_ode_states = n_ode_states

        self.state_init = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_ode_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.b_gain = nn.Parameter(torch.tensor(0.1))
        self.a_coeffs = nn.Parameter(
            init_companion_coeffs(
                n_ode_states=n_ode_states,
                pole_init=pole_init,
                resonant_init=resonant_init,
                alpha_init=alpha_init,
                xi_init=xi_init,
                omega0_init=omega0_init,
                sampling_time=sampling_time,
            )
        )

        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        x = self.state_init(y0)  # (batch, n_ode_states)

        outputs = []
        for t in range(T):
            u_t = u[:, t, :].squeeze(-1)
            if self.n_ode_states == 1:
                x = (self.b_gain * u_t + self.a_coeffs[0] * x[:, 0]).unsqueeze(-1)
            else:
                x_shift = x[:, 1:]
                x_last = self.b_gain * u_t + torch.sum(self.a_coeffs.unsqueeze(0) * x, dim=-1)
                x = torch.cat([x_shift, x_last.unsqueeze(-1)], dim=-1)

            x1 = x[:, 0:1]
            outputs.append(self.out_scale * torch.abs(x1) + self.out_bias)

        y_hat = torch.stack(outputs, dim=1)
        return y_hat, x


class HybridCEDModel(nn.Module):
    """
    Hybrid model: WienerCEDModel backbone + a small LSTM residual correction,
    intended to capture unmodeled effects (anti-aliasing filter dynamics per
    the Wiener-Hammerstein extension in eq. 3-5, sensor rectification
    imperfections, quantization noise from the pulse counter, ...).
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_ode_states: int = 3,
        n_hidden_states: int = 16,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 1,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        pole_init: float = 0.7,
        resonant_init: bool = False,
        alpha_init: float = 2.0,
        xi_init: float = 0.1,
        omega0_init: float = 25.0,
        sampling_time: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        self.physical = WienerCEDModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_ode_states=n_ode_states,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            pole_init=pole_init,
            resonant_init=resonant_init,
            alpha_init=alpha_init,
            xi_init=xi_init,
            omega0_init=omega0_init,
            sampling_time=sampling_time,
        )

        self.residual_block = RecurrentBlock(
            input_size=n_inputs,
            hidden_size=n_hidden_states,
            output_size=n_outputs,
            num_layers=num_layers,
            recurrent="LSTM",
            dropout_prob=dropout_prob,
        )

        self.residual_init_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, object]:
        y_phys, _ = self.physical(u, y0)

        y0_flat = y0.squeeze(1) if y0.dim() == 3 else y0
        h0 = self.residual_init_net(y0_flat).unsqueeze(0).repeat(
            self.residual_block.recurrent.num_layers, 1, 1
        )
        c0 = h0.clone()
        y_res, hidden = self.residual_block(u, (h0, c0))
        y_hat = y_phys + self.residual_scale * y_res
        return y_hat, hidden


def build_model_from_config(
    config_pars: dict,
    n_inputs: int,
    n_states: int,
    n_outputs: int,
) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return CEDRecurrentModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=hidden_sizes,
            num_layers=config_pars["num_layers"],
            recurrent=model_type,
            activation=activation,
            dropout_prob=dropout_prob,
            direct_feedthrough=config_pars.get("direct_feedthrough", False),
        )

    if model_type == "TCN":
        return CEDTCNModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            tcn_channels=config_pars.get("tcn_channels", 32),
            kernel_size=config_pars.get("kernel_size", 3),
            num_blocks=config_pars.get("num_blocks", 5),
            activation=activation,
            dropout_prob=dropout_prob,
            context_hidden_sizes=config_pars.get("context_hidden_sizes", hidden_sizes),
        )

    if model_type == "WIENER":
        return WienerCEDModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_ode_states=config_pars.get("n_ode_states", 3),
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            pole_init=config_pars.get("pole_init", 0.7),
            resonant_init=config_pars.get("resonant_init", False),
            alpha_init=config_pars.get("alpha_init", 2.0),
            xi_init=config_pars.get("xi_init", 0.1),
            omega0_init=config_pars.get("omega0_init", 25.0),
            sampling_time=config_pars.get("sampling_time", 0.02),
        )

    if model_type == "HYBRID":
        return HybridCEDModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_ode_states=config_pars.get("n_ode_states", 3),
            n_hidden_states=config_pars.get("n_hidden_states", 16),
            hidden_sizes=hidden_sizes,
            num_layers=config_pars.get("num_layers", 1),
            activation=activation,
            dropout_prob=dropout_prob,
            pole_init=config_pars.get("pole_init", 0.7),
            resonant_init=config_pars.get("resonant_init", False),
            alpha_init=config_pars.get("alpha_init", 2.0),
            xi_init=config_pars.get("xi_init", 0.1),
            omega0_init=config_pars.get("omega0_init", 25.0),
            sampling_time=config_pars.get("sampling_time", 0.02),
        )

    if model_type == "PHYSICS_TCN":
        return PhysicsGuidedTCNModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_ode_states=config_pars.get("n_ode_states", 3),
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            pole_init=config_pars.get("pole_init", 0.7),
            resonant_init=config_pars.get("resonant_init", False),
            alpha_init=config_pars.get("alpha_init", 2.0),
            xi_init=config_pars.get("xi_init", 0.1),
            omega0_init=config_pars.get("omega0_init", 25.0),
            sampling_time=config_pars.get("sampling_time", 0.02),
            tcn_channels=config_pars.get("tcn_channels", 32),
            kernel_size=config_pars.get("kernel_size", 3),
            num_blocks=config_pars.get("num_blocks", 5),
            context_hidden_sizes=config_pars.get("context_hidden_sizes"),
        )

    raise ValueError(
        "Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'WIENER', 'HYBRID', 'TCN', 'PHYSICS_TCN'}."
    )