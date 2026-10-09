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


class SilverboxRecurrentModel(nn.Module):
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


class SilverboxModel(nn.Module):
    """
    Grey-box model matching this benchmark's actual physical structure:
    an electronic implementation of a Duffing oscillator -- a 2nd-order
    LTI system with a 3rd-degree polynomial static nonlinearity in
    feedback:

        m*y'' + c*y' + k*y + k3*y^3 = u(t)

    States: x1 = y (output voltage), x2 = y' (its rate of change):

        x1[k+1] = x1[k] + e_pos * x2[k]
        x2[k+1] = a * x2[k] + b * x1[k] + b3 * x1[k]^3 + c_gain * u[k]

    Two changes from the most direct version of this model, both needed
    to avoid genuine numerical errors (not search/tuning choices):

    1. `a`/`b` are DERIVED each forward pass from a bounded pole
       magnitude and a free angle, instead of being learned directly.
       Learning `a`/`b` directly only checks stability at INIT -- nothing
       then stops gradient descent from later pushing them past the
       point where the recursion's poles exceed magnitude 1, causing an
       unpredictable, run-dependent NaN blowup partway through training
       (confirmed directly: identical code produced both a stable,
       converging run and a diverging one). Magnitude is parameterized
       via a `softplus`-constrained decay rate rather than a directly
       bounded value, so it can still represent an arbitrarily
       lightly-damped resonance without the vanishing-gradient
       saturation a hard sigmoid bound would cause near that regime.

    2. `c_gain` and `out_scale` are bounded via `softplus` rather than
       fully free scalars. Even with a stable pole, an unconstrained
       scale factor can still be pushed to a large enough value by
       training to overflow float32 -- confirmed as a second, separate
       cause of NaN, distinct from pole instability.

    Everything else -- the recursion structure itself, `e_pos` fixed as
    a simple constant, `b3` off by default, `state_init` zero-initialized
    -- is unchanged from the direct/classic form of this model.
    `pole_freq_hz_init`/`pole_damping_ratio_init`/`c_gain_init` are left
    at generic, non-benchmark-specific starting values; finding good
    values for these is exactly what `train.py`'s own search loop is for.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, hidden_sizes=None,
                 activation="Tanh", dropout_prob=0.0,
                 pole_freq_hz_init=50.0, pole_damping_ratio_init=0.1,
                 sampling_rate_hz=610.35, e_pos=0.1, c_gain_init=1.0, learn_b3=False,
                 b3_max=2.0, x_sat=3.0, state_clip=50.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if not (0.0 < pole_damping_ratio_init < 1.0):
            raise ValueError("pole_damping_ratio_init must lie in (0, 1) (fraction of critical damping).")

        self.state_init = FeedforwardNetwork(n_states, hidden_sizes, 2, activation=activation, dropout_prob=dropout_prob)
        nn.init.zeros_(self.state_init.net[-1].weight)
        nn.init.zeros_(self.state_init.net[-1].bias)

        self.dt = 1.0 / sampling_rate_hz
        self.e_pos = e_pos  # fixed constant, matching the classic model's own default

        omega_n = 2 * torch.pi * pole_freq_hz_init
        decay_rate_init = pole_damping_ratio_init * omega_n
        decay_rate_raw_init = torch.log(torch.expm1(torch.tensor(float(decay_rate_init))))
        self.decay_rate_raw = nn.Parameter(decay_rate_raw_init)

        angle_init = 2 * torch.pi * pole_freq_hz_init / sampling_rate_hz
        self.pole_angle = nn.Parameter(torch.tensor(float(angle_init)))

        c_gain_raw_init = torch.log(torch.expm1(torch.tensor(float(c_gain_init))))
        self.c_gain_raw = nn.Parameter(c_gain_raw_init)
        self.learn_b3 = learn_b3
        self.b3_max = float(b3_max)
        self.x_sat = float(x_sat)
        self.state_clip = float(state_clip)
        self.b3_raw = nn.Parameter(torch.tensor(0.0)) if learn_b3 else torch.tensor(0.0)

        out_scale_raw_init = torch.log(torch.expm1(torch.tensor(1.0)))
        self.out_scale_raw = nn.Parameter(out_scale_raw_init)
        self.out_bias = nn.Parameter(torch.zeros(1))

    def _compute_a_b(self):
        decay_rate = nn.functional.softplus(self.decay_rate_raw)
        magnitude = torch.exp(-decay_rate * self.dt)
        trace = 2 * magnitude * torch.cos(self.pole_angle)
        det = magnitude ** 2
        a = trace - 1.0
        b = (a - det) / self.e_pos
        return a, b

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        state = self.state_init(y0)  # (batch, 2): [y, y']
        a, b = self._compute_a_b()
        c_gain = nn.functional.softplus(self.c_gain_raw)
        out_scale = nn.functional.softplus(self.out_scale_raw)
        # b3 bounded via tanh, and the cubic term itself uses a tanh-saturated
        # x1 so |cubic_term| <= b3_max * x_sat**3 for ANY x1 -- needed because
        # this recursion is rolled forward over ~13000-sample uncropped
        # sequences for train/val metrics (5-7x longer than any BPTT window),
        # so an unbounded polynomial term can drift to overflow over that
        # horizon even when short training windows look fine. state_clip is
        # a generous last-resort float32 guard, not the primary mechanism.
        b3 = self.b3_max * torch.tanh(self.b3_raw) if self.learn_b3 else self.b3_raw

        outputs = []
        for t in range(T):
            u_t = u[:, t, :].squeeze(-1)
            x1, x2 = state[:, 0], state[:, 1]

            cubic_term = b3 * (self.x_sat * torch.tanh(x1 / self.x_sat)) ** 3
            x2_next = a * x2 + b * x1 + cubic_term + c_gain * u_t
            x1_next = x1 + self.e_pos * x2

            state = torch.clamp(torch.stack([x1_next, x2_next], dim=-1), min=-self.state_clip, max=self.state_clip)
            outputs.append((out_scale * x1_next + self.out_bias).unsqueeze(-1))

        y_hat = torch.stack(outputs, dim=1)
        return y_hat, state


class SilverboxNARXModel(nn.Module):
    """Pure feedforward NARX: predicts y[t] from a short lag window of past
    u/y plus the current u, autoregressive on its own past outputs during
    rollout. No recurrent hidden state at all -- a structurally different
    family from both the gated-RNN and physical-recursion models."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, hidden_sizes=None,
                 activation="ReLU", dropout_prob=0.0, narx_lag=10, out_clip=20.0, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        self.narx_lag = int(narx_lag)
        self.out_clip = float(out_clip)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        input_size = self.narx_lag * (n_inputs + n_outputs) + n_inputs
        self.net = FeedforwardNetwork(input_size, hidden_sizes, n_outputs, activation, dropout_prob)
        nn.init.zeros_(self.net.net[-1].weight)
        nn.init.zeros_(self.net.net[-1].bias)

    def forward(self, u, y0):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        batch_size, T, _ = u.shape
        half = y0.shape[-1] // 2
        u_buffer = y0[:, :self.narx_lag]
        y_buffer = y0[:, half:half + self.narx_lag]

        outputs = []
        for t in range(T):
            u_t = u[:, t, :]
            features = torch.cat([u_t, u_buffer, y_buffer], dim=-1)
            y_t = torch.clamp(self.net(features), min=-self.out_clip, max=self.out_clip)
            outputs.append(y_t.unsqueeze(1))
            u_buffer = torch.cat([u_t, u_buffer[:, :-1]], dim=-1)
            y_buffer = torch.cat([y_t, y_buffer[:, :-1]], dim=-1)
        y_hat = torch.cat(outputs, dim=1)
        return y_hat, None


class SilverboxHybridModel(nn.Module):
    """Grey+black-box hybrid: the same stabilized physical Duffing branch as
    SilverboxModel runs in parallel with an independent small GRU residual
    branch; outputs are summed. No feedback between branches -- the physical
    branch keeps its own proven stability (bounded cubic term, state clamp),
    and the GRU branch is long-horizon-stable by construction (gated
    activations bounded in [-1,1] regardless of what it learns), so it can
    pick up whatever the physics under-fits without inheriting the physical
    branch's optimization-landscape issues. The residual's output layer is
    zero-initialized so training starts from a pure-physics prediction and
    only turns on the residual where gradients indicate it helps -- avoiding
    PROGRAM.md's noted risk of a wild-scale random initial contribution
    destabilizing an otherwise well-behaved recursion.
    """

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, hidden_sizes=None,
                 activation="Tanh", dropout_prob=0.0,
                 pole_freq_hz_init=50.0, pole_damping_ratio_init=0.1,
                 sampling_rate_hz=610.35, e_pos=0.1, c_gain_init=1.0, learn_b3=True,
                 b3_max=2.0, x_sat=3.0, state_clip=50.0, residual_hidden=16, **kwargs):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [32, 32]
        if not (0.0 < pole_damping_ratio_init < 1.0):
            raise ValueError("pole_damping_ratio_init must lie in (0, 1) (fraction of critical damping).")

        self.state_init = FeedforwardNetwork(n_states, hidden_sizes, 2, activation=activation, dropout_prob=dropout_prob)
        nn.init.zeros_(self.state_init.net[-1].weight)
        nn.init.zeros_(self.state_init.net[-1].bias)

        self.dt = 1.0 / sampling_rate_hz
        self.e_pos = e_pos

        omega_n = 2 * torch.pi * pole_freq_hz_init
        decay_rate_init = pole_damping_ratio_init * omega_n
        decay_rate_raw_init = torch.log(torch.expm1(torch.tensor(float(decay_rate_init))))
        self.decay_rate_raw = nn.Parameter(decay_rate_raw_init)

        angle_init = 2 * torch.pi * pole_freq_hz_init / sampling_rate_hz
        self.pole_angle = nn.Parameter(torch.tensor(float(angle_init)))

        c_gain_raw_init = torch.log(torch.expm1(torch.tensor(float(c_gain_init))))
        self.c_gain_raw = nn.Parameter(c_gain_raw_init)
        self.learn_b3 = learn_b3
        self.b3_max = float(b3_max)
        self.x_sat = float(x_sat)
        self.state_clip = float(state_clip)
        self.b3_raw = nn.Parameter(torch.tensor(0.0)) if learn_b3 else torch.tensor(0.0)

        out_scale_raw_init = torch.log(torch.expm1(torch.tensor(1.0)))
        self.out_scale_raw = nn.Parameter(out_scale_raw_init)
        self.out_bias = nn.Parameter(torch.zeros(1))

        self.residual_init_net = FeedforwardNetwork(n_states, hidden_sizes, residual_hidden, activation, dropout_prob)
        self.residual_block = RecurrentBlock(n_inputs, residual_hidden, n_outputs, num_layers=1, recurrent="GRU", dropout_prob=dropout_prob)
        nn.init.zeros_(self.residual_block.output_layer.weight)
        nn.init.zeros_(self.residual_block.output_layer.bias)

    def get_param_groups(self, base_lr):
        physical_params = [self.decay_rate_raw, self.pole_angle, self.c_gain_raw, self.out_scale_raw, self.out_bias]
        if self.learn_b3:
            physical_params.append(self.b3_raw)
        physical_ids = {id(p) for p in physical_params}
        other_params = [p for p in self.parameters() if id(p) not in physical_ids]
        return [
            {"params": physical_params, "lr": base_lr * 10.0},
            {"params": other_params, "lr": base_lr},
        ]

    def _compute_a_b(self):
        decay_rate = nn.functional.softplus(self.decay_rate_raw)
        magnitude = torch.exp(-decay_rate * self.dt)
        trace = 2 * magnitude * torch.cos(self.pole_angle)
        det = magnitude ** 2
        a = trace - 1.0
        b = (a - det) / self.e_pos
        return a, b

    def forward(self, u: torch.Tensor, y0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y0.dim() == 3:
            y0 = y0.squeeze(1)

        batch_size, T, _ = u.shape
        state = self.state_init(y0)
        a, b = self._compute_a_b()
        c_gain = nn.functional.softplus(self.c_gain_raw)
        out_scale = nn.functional.softplus(self.out_scale_raw)
        b3 = self.b3_max * torch.tanh(self.b3_raw) if self.learn_b3 else self.b3_raw

        physical_outputs = []
        for t in range(T):
            u_t = u[:, t, :].squeeze(-1)
            x1, x2 = state[:, 0], state[:, 1]
            cubic_term = b3 * (self.x_sat * torch.tanh(x1 / self.x_sat)) ** 3
            x2_next = a * x2 + b * x1 + cubic_term + c_gain * u_t
            x1_next = x1 + self.e_pos * x2
            state = torch.clamp(torch.stack([x1_next, x2_next], dim=-1), min=-self.state_clip, max=self.state_clip)
            physical_outputs.append((out_scale * x1_next + self.out_bias).unsqueeze(-1))
        y_phys = torch.stack(physical_outputs, dim=1)

        residual_hidden0 = self.residual_init_net(y0).unsqueeze(0)
        y_res, _ = self.residual_block(u, residual_hidden0)

        return y_phys + y_res, state


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)

    def forward(self, x):
        out = self.conv(x)
        return out[..., :-self.padding] if self.padding > 0 else out


class SilverboxTCNModel(nn.Module):
    """Temporal convolutional network: a stack of dilated causal 1D
    convolutions (exponentially growing dilation -> a bounded, linear-in-T
    receptive field, unlike full self-attention's quadratic cost -- relevant
    here because train.py's own metric evaluation rolls every model forward
    over ~13000-sample uncropped sequences). No recurrent hidden state, no
    polynomial feedback term -- a genuinely different family (parallelizable
    convolutional receptive field instead of a step-by-step recursion). The
    initial-condition vector y0 is mapped to a small context embedding and
    concatenated to the input at every timestep (a static conditioning
    signal, since a pure conv stack has no persistent state of its own to
    carry y0 forward implicitly)."""

    def __init__(self, n_inputs=1, n_states=1, n_outputs=1, hidden_sizes=None,
                 activation="ReLU", dropout_prob=0.0, tcn_channels=32, tcn_layers=6,
                 kernel_size=3, context_dim=16, **kwargs):
        super().__init__()
        self.context_net = nn.Sequential(nn.Linear(n_states, context_dim), make_activation(activation))
        blocks: list[nn.Module] = []
        channels = n_inputs + context_dim
        for layer_index in range(tcn_layers):
            dilation = 2 ** layer_index
            blocks.append(CausalConv1d(channels, tcn_channels, kernel_size, dilation))
            blocks.append(make_activation(activation))
            if dropout_prob > 0.0:
                blocks.append(nn.Dropout(dropout_prob))
            channels = tcn_channels
        self.tcn = nn.Sequential(*blocks)
        self.output_layer = nn.Conv1d(channels, n_outputs, kernel_size=1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, u, y0):
        if y0.dim() == 3:
            y0 = y0.squeeze(1)
        batch_size, T, _ = u.shape
        context = self.context_net(y0).unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([u, context], dim=-1).transpose(1, 2)
        features = self.tcn(x)
        y_hat = self.output_layer(features).transpose(1, 2)
        return y_hat, None


def build_model_from_config(config_pars: dict, n_inputs: int, n_states: int, n_outputs: int) -> nn.Module:
    model_type = str(config_pars.get("type", "LSTM")).upper()
    hidden_sizes = list(config_pars["hidden_sizes"])
    activation = config_pars.get("activation", "ReLU")
    dropout_prob = config_pars.get("dropout_prob", 0.0)

    if model_type in {"RNN", "GRU", "LSTM"}:
        return SilverboxRecurrentModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            n_hidden_states=config_pars["n_hidden_states"], hidden_sizes=hidden_sizes,
            num_layers=config_pars["num_layers"], recurrent=model_type, activation=activation,
            dropout_prob=dropout_prob, direct_feedthrough=config_pars.get("direct_feedthrough", False),
            rnn_nonlinearity=config_pars.get("rnn_nonlinearity", "tanh"),
        )

    if model_type == "SILVERBOX":
        return SilverboxModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
            pole_freq_hz_init=config_pars.get("pole_freq_hz_init", 50.0),
            pole_damping_ratio_init=config_pars.get("pole_damping_ratio_init", 0.1),
            sampling_rate_hz=config_pars.get("sampling_rate_hz", 610.35),
            e_pos=config_pars.get("e_pos", 0.1),
            c_gain_init=config_pars.get("c_gain_init", 1.0),
            learn_b3=config_pars.get("learn_b3", False),
            b3_max=config_pars.get("b3_max", 2.0),
            x_sat=config_pars.get("x_sat", 3.0),
            state_clip=config_pars.get("state_clip", 50.0),
        )

    if model_type == "NARX":
        return SilverboxNARXModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
            narx_lag=config_pars.get("narx_lag", 10), out_clip=config_pars.get("out_clip", 20.0),
        )

    if model_type == "SILVERBOX_HYBRID":
        return SilverboxHybridModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            hidden_sizes=hidden_sizes, activation=activation, dropout_prob=dropout_prob,
            pole_freq_hz_init=config_pars.get("pole_freq_hz_init", 50.0),
            pole_damping_ratio_init=config_pars.get("pole_damping_ratio_init", 0.1),
            sampling_rate_hz=config_pars.get("sampling_rate_hz", 610.35),
            e_pos=config_pars.get("e_pos", 0.1),
            c_gain_init=config_pars.get("c_gain_init", 1.0),
            learn_b3=config_pars.get("learn_b3", True),
            b3_max=config_pars.get("b3_max", 2.0),
            x_sat=config_pars.get("x_sat", 3.0),
            state_clip=config_pars.get("state_clip", 50.0),
            residual_hidden=config_pars.get("residual_hidden", 16),
        )

    if model_type == "TCN":
        return SilverboxTCNModel(
            n_inputs=n_inputs, n_states=n_states, n_outputs=n_outputs,
            activation=activation, dropout_prob=dropout_prob,
            tcn_channels=config_pars.get("tcn_channels", 32), tcn_layers=config_pars.get("tcn_layers", 6),
            kernel_size=config_pars.get("kernel_size", 3), context_dim=config_pars.get("context_dim", 16),
        )

    raise ValueError("Unsupported model type. Expected one of {'RNN', 'GRU', 'LSTM', 'SILVERBOX', 'SILVERBOX_HYBRID', 'NARX', 'TCN'}.")