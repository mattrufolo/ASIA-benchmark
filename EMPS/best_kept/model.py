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


# ---------------------------------------------------------------------------------
# BLACK-BOX family: generic recurrent sequence models. No knowledge of the EMPS
# physics is used; the initial hidden state is predicted from the initial-condition
# vector y0 (past inputs/outputs), and the recurrent network maps the force input u(t)
# to the position output q(t) directly.
# ---------------------------------------------------------------------------------
class EMPSBlackBoxModel(nn.Module):
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


class ClosedFormCTC(nn.Module):
    """
    Closed-form Continuous-time network (CfC) -- a second black-box family, useful
    as an architecturally different alternative to the LSTM/GRU/RNN family above.
    Each neuron has a learnable time constant tau_i > 0.
    State update: h[t+1] = A_i * h[t] + (1-A_i) * f(h[t], u[t])
    where A_i = exp(-dt/tau_i) in (0,1) -- unconditionally stable.
    `dt` is the known EMPS sampling time (1 ms), passed in fixed (not learned).
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_hidden_states: int = 64,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        direct_feedthrough: bool = False,
        dt: float = 0.001,
        tau_max: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.n_hidden = n_hidden_states
        self.direct_feedthrough = direct_feedthrough

        # Initialize tau so neurons span [dt, tau_max seconds] in log space.
        log_tau_init = torch.linspace(
            torch.log(torch.tensor(dt)),
            torch.log(torch.tensor(tau_max)),
            n_hidden_states,
        )
        self.log_tau = nn.Parameter(log_tau_init)
        self.register_buffer("dt", torch.tensor(dt, dtype=torch.float32))

        self.f_net = FeedforwardNetwork(
            input_size=n_hidden_states + n_inputs,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.ic_net = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=n_hidden_states,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        readout_in = n_hidden_states + n_inputs if direct_feedthrough else n_hidden_states
        self.readout = nn.Linear(readout_in, n_outputs)

    def forward(
        self, u: torch.Tensor, y0: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch, T, _ = u.shape
        h = self.ic_net(y0)

        tau = torch.exp(self.log_tau)
        A = torch.exp(-self.dt / tau).unsqueeze(0)  # (1, n_hidden)

        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            h = A * h + (1.0 - A) * self.f_net(torch.cat([h, u_t], dim=-1))
            if self.direct_feedthrough:
                y_t = self.readout(torch.cat([h, u_t], dim=-1))
            else:
                y_t = self.readout(h)
            outputs.append(y_t.unsqueeze(1))

        return torch.cat(outputs, dim=1), h


# ---------------------------------------------------------------------------------
# WHITE-BOX family: directly inspired by the EMPS paper's Direct Dynamic Model, eq.(4):
#     qdd(t) = tau(t)/M - (Fv/M)*qd(t) - (Fc/M)*sign(qd(t)) - offset/M
# Since the benchmark input `vir` already plays the role of tau(t) (force referred to
# the load side), this is directly a 2nd-order ODE in state [q, qd] driven by u = vir.
# All signals/states are handled in *normalized* space (like the raw inputs/outputs),
# so M, Fv, Fc, offset below are effectively normalized-space parameters, not literal
# SI values -- exactly as in the reference cascaded-tank grey-box model.
#
# `dt` is passed in as the true (known) EMPS sampling time and is NOT learned, since
# unlike the tank benchmark, here the sampling time is known exactly (1 ms).
# ---------------------------------------------------------------------------------
class PhysicalEMPSModel(nn.Module):
    """Grey/white-box model with symmetric Coulomb + viscous friction (paper eq. 1/4)."""

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        dt: float = 0.001,
        integrator: str = "euler",
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        self.integrator = integrator.lower()
        self.register_buffer("dt", torch.tensor(float(dt), dtype=torch.float32))

        # Predicts the initial [q, qd] (normalized space) from the initial-condition
        # vector y0 (past inputs/outputs).
        self.state_init = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=2,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        # Log-parameterize M, Fv, Fc to keep them positive, as in the paper's IDM.
        self.log_M = nn.Parameter(torch.zeros(1))
        self.log_Fv = nn.Parameter(torch.zeros(1))
        self.log_Fc = nn.Parameter(torch.zeros(1))
        self.offset = nn.Parameter(torch.zeros(1))

        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    def _accel(self, q: torch.Tensor, qd: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        M = torch.exp(self.log_M)
        Fv = torch.exp(self.log_Fv)
        Fc = torch.exp(self.log_Fc)
        friction = Fv * qd + Fc * torch.tanh(50.0 * qd)  # smoothed sign(qd) for a usable gradient
        return (u_t - friction - self.offset) / M

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        z_init = self.state_init(y0)
        q = z_init[:, 0:1]
        qd = z_init[:, 1:2]
        dt = self.dt

        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            if self.integrator == "rk4":
                a1 = self._accel(q, qd, u_t)
                q1, qd1 = q + 0.5 * dt * qd, qd + 0.5 * dt * a1
                a2 = self._accel(q1, qd1, u_t)
                q2, qd2 = q + 0.5 * dt * qd1, qd + 0.5 * dt * a2
                a3 = self._accel(q2, qd2, u_t)
                q3, qd3 = q + dt * qd2, qd + dt * a3
                a4 = self._accel(q3, qd3, u_t)
                q = q + (dt / 6.0) * (qd + 2 * qd1 + 2 * qd2 + qd3)
                qd = qd + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
            else:  # euler (default, matches the paper's DDM most directly)
                qdd = self._accel(q, qd, u_t)
                qd = qd + dt * qdd
                q = q + dt * qd
            outputs.append(self.out_scale * q + self.out_bias)

        y_hat = torch.stack(outputs, dim=1)
        return y_hat, (q, qd)


class PhysicalEMPSAsymFrictionModel(nn.Module):
    """
    White-box model with the ASYMMETRIC friction reported in (Janot et al., 2017) and
    summarized in the paper, eq. (12):
        tau_fric = Fv+ * 0+(qd) + Fc+ * sign(0+(qd)) + Fv- * 0-(qd) + Fc- * sign(0-(qd))
    with 0+(qd) = qd*(1+sign(qd))/2 and 0-(qd) = qd*(1-sign(qd))/2.
    """

    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        hidden_sizes: list[int] | None = None,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        dt: float = 0.001,
        integrator: str = "euler",
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        self.integrator = integrator.lower()
        self.register_buffer("dt", torch.tensor(float(dt), dtype=torch.float32))

        self.state_init = FeedforwardNetwork(
            input_size=n_states,
            hidden_sizes=hidden_sizes,
            output_size=2,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        self.log_M = nn.Parameter(torch.zeros(1))
        self.log_Fv_p = nn.Parameter(torch.zeros(1))
        self.log_Fc_p = nn.Parameter(torch.zeros(1))
        self.log_Fv_m = nn.Parameter(torch.zeros(1))
        self.log_Fc_m = nn.Parameter(torch.zeros(1))
        self.offset = nn.Parameter(torch.zeros(1))

        self.out_scale = nn.Parameter(torch.ones(1))
        self.out_bias = nn.Parameter(torch.zeros(1))

    def _accel(self, q: torch.Tensor, qd: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        M = torch.exp(self.log_M)
        Fv_p, Fc_p = torch.exp(self.log_Fv_p), torch.exp(self.log_Fc_p)
        Fv_m, Fc_m = torch.exp(self.log_Fv_m), torch.exp(self.log_Fc_m)

        qd_pos = F.relu(qd)  # smooth analogue of qd * (1+sign(qd))/2
        qd_neg = -F.relu(-qd)  # smooth analogue of qd * (1-sign(qd))/2 (<= 0)

        # tanh(50*.) gives a smooth, well-conditioned analogue of sign(.) at the branch.
        friction = (
            Fv_p * qd_pos + Fc_p * torch.tanh(50.0 * qd_pos)
            + Fv_m * qd_neg + Fc_m * torch.tanh(50.0 * qd_neg)
        )
        return (u_t - friction - self.offset) / M

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        z_init = self.state_init(y0)
        q = z_init[:, 0:1]
        qd = z_init[:, 1:2]
        dt = self.dt

        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            if self.integrator == "rk4":
                a1 = self._accel(q, qd, u_t)
                q1, qd1 = q + 0.5 * dt * qd, qd + 0.5 * dt * a1
                a2 = self._accel(q1, qd1, u_t)
                q2, qd2 = q + 0.5 * dt * qd1, qd + 0.5 * dt * a2
                a3 = self._accel(q2, qd2, u_t)
                q3, qd3 = q + dt * qd2, qd + dt * a3
                a4 = self._accel(q3, qd3, u_t)
                q = q + (dt / 6.0) * (qd + 2 * qd1 + 2 * qd2 + qd3)
                qd = qd + (dt / 6.0) * (a1 + 2 * a2 + 2 * a3 + a4)
            else:
                qdd = self._accel(q, qd, u_t)
                qd = qd + dt * qdd
                q = q + dt * qd
            outputs.append(self.out_scale * q + self.out_bias)

        y_hat = torch.stack(outputs, dim=1)
        return y_hat, (q, qd)


# ---------------------------------------------------------------------------------
# GREY-BOX family: the physical (symmetric or asymmetric friction) backbone above,
# plus a small LSTM residual correction that can pick up unmodeled effects (e.g. the
# nested PD loop's imperfect tracking, encoder quantization, mechanical resonances).
# ---------------------------------------------------------------------------------
class HybridEMPSModel(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_states: int,
        n_outputs: int,
        n_hidden_states: int = 16,
        hidden_sizes: list[int] | None = None,
        num_layers: int = 1,
        activation: str = "Tanh",
        dropout_prob: float = 0.0,
        dt: float = 0.001,
        integrator: str = "euler",
        asymmetric_friction: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [32, 32]

        physical_cls = PhysicalEMPSAsymFrictionModel if asymmetric_friction else PhysicalEMPSModel
        self.physical = physical_cls(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            dt=dt,
            integrator=integrator,
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

    def forward(
        self, u: torch.Tensor, y0: torch.Tensor
    ) -> tuple[torch.Tensor, object]:
        y_phys, _ = self.physical(u, y0)

        if y0.dim() == 3:
            y0_flat = y0.squeeze(1)
        else:
            y0_flat = y0
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
    dt: float = 0.001,
) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return EMPSBlackBoxModel(
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

    if model_type == "LTC":
        return ClosedFormCTC(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"],
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            direct_feedthrough=config_pars.get("direct_feedthrough", False),
            dt=dt,
        )

    if model_type == "PHYSICAL":
        return PhysicalEMPSModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            dt=dt,
            integrator=config_pars.get("integrator", "euler"),
        )

    if model_type == "PHYSICAL_ASYM":
        return PhysicalEMPSAsymFrictionModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            hidden_sizes=hidden_sizes,
            activation=activation,
            dropout_prob=dropout_prob,
            dt=dt,
            integrator=config_pars.get("integrator", "euler"),
        )

    if model_type == "HYBRID":
        return HybridEMPSModel(
            n_inputs=n_inputs,
            n_states=n_states,
            n_outputs=n_outputs,
            n_hidden_states=config_pars.get("n_hidden_states", 16),
            hidden_sizes=hidden_sizes,
            num_layers=config_pars.get("num_layers", 1),
            activation=activation,
            dropout_prob=dropout_prob,
            dt=dt,
            integrator=config_pars.get("integrator", "euler"),
            asymmetric_friction=config_pars.get("asymmetric_friction", False),
        )

    raise ValueError(
        "Unsupported model type. Expected one of "
        "{'RNN', 'GRU', 'LSTM', 'LTC', 'PHYSICAL', 'PHYSICAL_ASYM', 'HYBRID'}."
    )
