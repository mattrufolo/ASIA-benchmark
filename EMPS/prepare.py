from __future__ import annotations

import json
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import nonlinear_benchmarks
except ImportError as exc:
    raise SystemExit(
        "Missing dependency `nonlinear_benchmarks`. Install the benchmark requirements first."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR / "data"
DOWNLOAD_DIR = RAW_DATA_DIR / "downloads"

# EMPS (Electro-Mechanical Positioning System): a pure integrator (position
# = integral of velocity, no restoring/spring term) driven by a motor
# force, with viscous + Coulomb friction. nonlinear_benchmarks.EMPS()
# returns ONE train trajectory (DATA_EMPS.mat) and ONE, SEPARATE, test
# trajectory (DATA_EMPS_PULSES.mat, a genuinely different experiment, not
# a continuation of train).
#
# --------------------------------------------------------------------------
# VALIDATION DESIGN: a single 80/20 split of the TRAIN trajectory (no
# k-fold CV). The key design point is HOW validation is scored:
#   - Training windows are sampled only from the FIRST 80% of the train
#     trajectory (random short crops -- see sample_training_window()).
#   - Validation runs the model CONTINUOUSLY from t=0 through the ENTIRE
#     train trajectory (all 100%), but RMSE is computed only over the
#     LAST 20% (via this sequence's own `warmup` field).
# This matters because each training window gets its initial condition
# (q0, qd0 equivalent) built from the REAL ground-truth u/y history right
# before it -- i.e. every training window is re-anchored to the true
# state, so a small systematic model bias (e.g. a slightly wrong physical
# parameter) barely shows up in a short window's loss, even though the
# SAME bias compounds badly once a model runs continuously, carrying its
# own (possibly biased) state forward with no resets -- exactly what
# validation and the official test do. Scoring validation as one
# continuous run (not more short windows) is what actually exposes that
# gap to the search, instead of only ever showing up later as a
# monitoring-only number from test.py.
# --------------------------------------------------------------------------

input_names = ["motor_force"]
output_names = ["load_position"]

config_pars_general = {
    "benchmark_name": "EMPS",
    "device": "cpu",
    "seed": 42,
    "history_window": 20,  # matches the official test's own state_initialization_window_length
    "benchmark_test_initialization_window_length": 20,  # recomputed from the live benchmark value in main()
    "warmup_test": 0,  # recomputed in main()/loaders from the benchmark value
    # Last 20% of the (post-warmup) train trajectory is reserved for
    # validation SCORING; the first 80% is used for training windows.
    "validation_fraction": 0.2,
    "window_min_length": 500,
    "extend_windows_to_range_end": False,
    "crops_per_step": 4,
    "validation_metric": "rmse",
    "base_path": "./cached_data",
    "raw_data_path": "./data",
    "checkpoint_path": "./checkpoints",
    "plots_path": "./plots",
    "log_dir": "./logs",
    "eval_every": 25,
    # Per-iteration wall-clock safety net, NOT early stopping (see train.py:
    # training always runs the fixed config_pars["max_epochs"], full stop --
    # this is just a ceiling so a single ASIA iteration can't run forever).
    "time_budget_seconds": 1800.0,
    "n_inputs": 1,
    "n_outputs": 1,
    "n_states": 40,  # recomputed below (2 * history_window, given n_inputs=n_outputs=1)
}


@dataclass
class EMPSSequence:
    name: str
    u: torch.Tensor
    y: torch.Tensor
    y0: torch.Tensor
    sampling_time: float
    warmup: int = 0
    start_index: int = 0
    stop_index: int = 0

    @property
    def num_samples(self) -> int:
        return int(self.u.shape[1])

    def clone(self) -> "EMPSSequence":
        return EMPSSequence(
            name=self.name, u=self.u.clone(), y=self.y.clone(), y0=self.y0.clone(),
            sampling_time=self.sampling_time, warmup=self.warmup,
            start_index=self.start_index, stop_index=self.stop_index,
        )


class Normalizer:
    def __init__(self, u_mean, u_std, y_mean, y_std, history_window, n_inputs, n_outputs, eps=1e-6):
        self.u_mean = u_mean.float()
        self.u_std = torch.clamp(u_std.float(), min=eps)
        self.y_mean = y_mean.float()
        self.y_std = torch.clamp(y_std.float(), min=eps)
        self.history_window = int(history_window)
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.eps = eps

    @classmethod
    def fit(cls, sequences: list[EMPSSequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one sequence is required to fit the normalizer.")
        u_all = torch.cat([s.u.reshape(-1, s.u.shape[-1]) for s in sequences], dim=0)
        y_all = torch.cat([s.y.reshape(-1, s.y.shape[-1]) for s in sequences], dim=0)
        return cls(
            u_mean=u_all.mean(dim=0, keepdim=True), u_std=u_all.std(dim=0, keepdim=True, unbiased=False),
            y_mean=y_all.mean(dim=0, keepdim=True), y_std=y_all.std(dim=0, keepdim=True, unbiased=False),
            history_window=history_window, n_inputs=int(sequences[0].u.shape[-1]), n_outputs=int(sequences[0].y.shape[-1]), eps=eps,
        )

    def normalize_sequence(self, sequence: EMPSSequence) -> EMPSSequence:
        normalized = sequence.clone()
        normalized.u = (normalized.u - self.u_mean.unsqueeze(0)) / self.u_std.unsqueeze(0)
        normalized.y = (normalized.y - self.y_mean.unsqueeze(0)) / self.y_std.unsqueeze(0)

        u_hist_dim = self.history_window * self.n_inputs
        y_hist_dim = self.history_window * self.n_outputs
        ic_mean = torch.cat([self.u_mean.repeat(1, self.history_window), self.y_mean.repeat(1, self.history_window)], dim=1).unsqueeze(0)
        ic_std = torch.cat([self.u_std.repeat(1, self.history_window), self.y_std.repeat(1, self.history_window)], dim=1).unsqueeze(0)
        if sequence.y0.shape[-1] != u_hist_dim + y_hist_dim:
            raise ValueError("Initial-condition vector size is inconsistent with history_window.")
        normalized.y0 = (normalized.y0 - ic_mean) / ic_std
        return normalized

    def denormalize_y_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        std = self.y_std.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        return values * std + mean

    def state_dict(self):
        return {
            "u_mean": self.u_mean.detach().cpu().numpy(), "u_std": self.u_std.detach().cpu().numpy(),
            "y_mean": self.y_mean.detach().cpu().numpy(), "y_std": self.y_std.detach().cpu().numpy(),
            "history_window": int(self.history_window), "n_inputs": int(self.n_inputs),
            "n_outputs": int(self.n_outputs), "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state) -> "Normalizer":
        return cls(
            u_mean=torch.as_tensor(state["u_mean"], dtype=torch.float32), u_std=torch.as_tensor(state["u_std"], dtype=torch.float32),
            y_mean=torch.as_tensor(state["y_mean"], dtype=torch.float32), y_std=torch.as_tensor(state["y_std"], dtype=torch.float32),
            history_window=int(state["history_window"]), n_inputs=int(state["n_inputs"]),
            n_outputs=int(state["n_outputs"]), eps=float(state.get("eps", 1e-6)),
        )


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic)
    except Exception:
        pass


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_official_splits(force_download: bool = False):
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    train_tuple, test_tuple = nonlinear_benchmarks.EMPS(
        atleast_2d=True, always_return_tuples_of_datasets=True,
        dir_placement=str(DOWNLOAD_DIR), force_download=force_download,
    )
    return train_tuple[0], test_tuple[0]


def benchmark_to_arrays(dataset) -> tuple[np.ndarray, np.ndarray, float]:
    u = np.asarray(dataset.u, dtype=np.float32)
    y = np.asarray(dataset.y, dtype=np.float32)
    if u.ndim == 1:
        u = u[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return u, y, float(dataset.sampling_time)


def build_initial_condition(u_full, y_full, start_index, history_window) -> np.ndarray:
    if start_index < history_window:
        raise ValueError("Not enough past samples to build the initial-condition vector.")
    u_history = [u_full[start_index - lag] for lag in range(1, history_window + 1)]
    y_history = [y_full[start_index - lag] for lag in range(1, history_window + 1)]
    ic_vector = np.concatenate([np.concatenate(u_history, axis=0), np.concatenate(y_history, axis=0)], axis=0)
    return ic_vector.astype(np.float32)


def effective_test_warmup(history_window: int, benchmark_window_length: int) -> int:
    return max(0, int(benchmark_window_length) - int(history_window))


def make_sequence(name, u_full, y_full, sampling_time, start_index, stop_index, history_window, warmup=0) -> EMPSSequence:
    y0_vector = build_initial_condition(u_full, y_full, start_index, history_window)
    return EMPSSequence(
        name=name, u=torch.from_numpy(u_full[start_index:stop_index]).unsqueeze(0),
        y=torch.from_numpy(y_full[start_index:stop_index]).unsqueeze(0),
        y0=torch.from_numpy(y0_vector).view(1, 1, -1), sampling_time=sampling_time,
        warmup=warmup, start_index=start_index, stop_index=stop_index,
    )


def load_raw_train_arrays() -> tuple[np.ndarray, np.ndarray, float]:
    path = RAW_DATA_DIR / "train_full_raw.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `prepare.py` first.")
    data = np.load(path)
    return np.asarray(data["u"], dtype=np.float32), np.asarray(data["y"], dtype=np.float32), float(data["sampling_time"])


def sample_training_window(
    u_full, y_full, sampling_time, history_window, max_index, min_window_length, rng, extend_to_range_end=False
) -> EMPSSequence:
    """Samples a random training window from [history_window, max_index) --
    i.e. only the FIRST 80% of the train trajectory (max_index is the
    80% split point). Never draws from the held-out validation tail.

    `min_window_length` doubles as a "force a specific length" knob: pass
    `min_window_length = max_index - history_window` to always get the
    entire first-80% span as a single window (this is how train.py's
    optional `long_window_prob` mixes in occasional full-length windows
    without needing any change here)."""
    lo = history_window
    hi = max_index
    max_start = hi - min_window_length
    if max_start < lo:
        start_index = lo
        window_length = max(1, hi - lo)
    elif extend_to_range_end:
        start_index = int(rng.integers(lo, max_start + 1))
        window_length = hi - start_index
    else:
        start_index = int(rng.integers(lo, max_start + 1))
        max_window_length = hi - start_index
        window_length = int(rng.integers(min_window_length, max_window_length + 1))
    stop = start_index + window_length
    return make_sequence(f"window_{start_index}_{stop}", u_full, y_full, sampling_time, start_index, stop, history_window, warmup=0)


def save_sequence(path: Path, sequence: EMPSSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, name=np.asarray(sequence.name), u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32), y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time), warmup=np.int64(sequence.warmup),
        start_index=np.int64(sequence.start_index), stop_index=np.int64(sequence.stop_index),
    )


def load_sequence(path: Path) -> EMPSSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")
    with np.load(path, allow_pickle=True) as data:
        name_value = data["name"]
        name = name_value.item() if np.asarray(name_value).shape == () else str(name_value)
        return EMPSSequence(
            name=str(name), u=torch.from_numpy(np.asarray(data["u"], dtype=np.float32)),
            y=torch.from_numpy(np.asarray(data["y"], dtype=np.float32)), y0=torch.from_numpy(np.asarray(data["y0"], dtype=np.float32)),
            sampling_time=float(data["sampling_time"]), warmup=int(data["warmup"]),
            start_index=int(data.get("start_index", 0)), stop_index=int(data.get("stop_index", 0)),
        )


def save_raw_arrays(path: Path, u, y, sampling_time) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, u=u.astype(np.float32), y=y.astype(np.float32), sampling_time=np.float32(sampling_time))


def save_config(config: dict) -> None:
    base_path = Path(config["base_path"])
    base_path.mkdir(parents=True, exist_ok=True)
    with (base_path / "config_params_general.pkl").open("wb") as handle:
        pickle.dump(config, handle)


def load_config(base_path=None) -> dict:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    path = root / "config_params_general.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached config: {path}. Run `prepare.py` first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _finalize_config(config: dict, reference_sequence: EMPSSequence) -> dict:
    config["n_inputs"] = int(reference_sequence.u.shape[-1])
    config["n_outputs"] = int(reference_sequence.y.shape[-1])
    config["n_states"] = int(reference_sequence.y0.shape[-1])
    benchmark_window_length = int(config.get("benchmark_test_initialization_window_length", config_pars_general["benchmark_test_initialization_window_length"]))
    config["benchmark_test_initialization_window_length"] = benchmark_window_length
    config["warmup_test"] = effective_test_warmup(int(config["history_window"]), benchmark_window_length)
    return config


def load_validation_sequence(base_path=None) -> EMPSSequence:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return load_sequence(root / "validation_sequence.npz")


def load_test_sequence(base_path=None) -> EMPSSequence:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return load_sequence(root / "test_sequence.npz")


def load_train_config(base_path=None) -> dict:
    """Returns the general config plus `train_split_index`, the index (in
    the raw train array) marking the 80% boundary -- training windows must
    never be sampled past this point."""
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    config = load_config(root)
    validation_sequence = load_validation_sequence(root)
    config = _finalize_config(config, validation_sequence)
    config["train_split_index"] = validation_sequence.start_index + validation_sequence.warmup
    return config


def load_datasets_and_config(base_path=None) -> tuple[EMPSSequence, EMPSSequence, dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    validation_sequence = load_validation_sequence(root)
    test_sequence = load_test_sequence(root)
    config = load_config(root)
    config = _finalize_config(config, validation_sequence)
    config["train_split_index"] = validation_sequence.start_index + validation_sequence.warmup
    test_sequence.warmup = int(config["warmup_test"])
    return validation_sequence, test_sequence, config


def plot_full_trajectory(name, u, y, sampling_time, plot_path: Path, split_time: float | None = None) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(len(u), dtype=np.float32) * sampling_time
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_axis, u[:, 0], linewidth=0.5, color="#006d77")
    axes[0].set_ylabel(input_names[0] + " [N]")
    axes[0].set_title(f"{name} input trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time_axis, y[:, 0], linewidth=0.5, color="#bc6c25")
    axes[1].set_ylabel(output_names[0] + " [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title(f"{name} output trajectory")
    axes[1].grid(True, alpha=0.25)
    if split_time is not None:
        for ax in axes:
            ax.axvline(split_time, color="black", linestyle="--", linewidth=1.0, label="80/20 split")
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_metadata(config, validation_sequence, test_sequence) -> None:
    base_path = Path(config["base_path"])
    metadata = {
        "benchmark_name": config["benchmark_name"], "sampling_time": validation_sequence.sampling_time,
        "history_window": config["history_window"], "validation_fraction": config["validation_fraction"],
        "train_split_index": validation_sequence.start_index + validation_sequence.warmup,
        "validation_sequence": {"num_samples": validation_sequence.num_samples, "warmup": validation_sequence.warmup},
        "test_sequence": {"num_samples": test_sequence.num_samples, "warmup": test_sequence.warmup},
        "notes": [
            "SINGLE 80/20 split of the train trajectory (DATA_EMPS.mat). "
            "Training windows (random short crops) are sampled ONLY from "
            "the first 80%. Validation runs the model CONTINUOUSLY from "
            "t=0 through the ENTIRE train trajectory, but RMSE is computed "
            "ONLY over the last 20% (via this sequence's own `warmup` "
            "field, reusing the same mechanism the official test's "
            "warmup already uses).",
            "The official test trajectory (DATA_EMPS_PULSES.mat) is a "
            "genuinely separate experiment -- evaluated the same way "
            "(continuous simulation, RMSE in mm after the official "
            "warmup).",
            "No ensembling: a single model is trained and evaluated "
            "directly.",
        ],
    }
    (base_path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    set_global_seed(config_pars_general["seed"])
    base_path = Path(config_pars_general["base_path"])
    raw_data_path = Path(config_pars_general["raw_data_path"])
    plots_path = Path(config_pars_general["plots_path"])
    checkpoint_path = Path(config_pars_general["checkpoint_path"])
    log_dir = Path(config_pars_general["log_dir"])

    for path in (plots_path, checkpoint_path, log_dir):
        if path.exists():
            shutil.rmtree(path)
    for path in (base_path, raw_data_path, plots_path, checkpoint_path, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_official_splits(force_download=False)
    u_train_full, y_train_full, sampling_time = benchmark_to_arrays(train_raw.atleast_2d())
    u_test_full, y_test_full, sampling_time_test = benchmark_to_arrays(test_raw.atleast_2d())

    history_window = int(config_pars_general["history_window"])
    validation_fraction = float(config_pars_general["validation_fraction"])
    benchmark_window_length = int(getattr(test_raw, "state_initialization_window_length", 20))
    config_pars_general["benchmark_test_initialization_window_length"] = benchmark_window_length
    config_pars_general["warmup_test"] = effective_test_warmup(history_window, benchmark_window_length)

    # 80/20 split of the USABLE (post-history-window) train trajectory.
    usable_length = len(u_train_full) - history_window
    validation_tail_length = int(round(usable_length * validation_fraction))
    train_split_index = history_window + (usable_length - validation_tail_length)

    # The validation sequence spans the ENTIRE train trajectory from
    # t=history_window to the end -- the model must simulate continuously
    # through the first 80% before reaching the scored last 20%. `warmup`
    # marks how many samples (from this sequence's own start) are NOT
    # counted in RMSE -- i.e. the first 80% -- reusing the exact same
    # mechanism the official test's warmup already uses.
    validation_warmup = train_split_index - history_window
    validation_sequence = make_sequence(
        "validation_full_horizon", u_train_full, y_train_full, sampling_time,
        history_window, len(u_train_full), history_window, warmup=validation_warmup,
    )
    save_sequence(base_path / "validation_sequence.npz", validation_sequence)

    test_sequence = make_sequence(
        "test", u_test_full, y_test_full, sampling_time_test, history_window, len(u_test_full),
        history_window, warmup=config_pars_general["warmup_test"],
    )
    save_sequence(base_path / "test_sequence.npz", test_sequence)

    save_raw_arrays(raw_data_path / "train_full_raw.npz", u_train_full, y_train_full, sampling_time)
    save_raw_arrays(raw_data_path / "test_full_raw.npz", u_test_full, y_test_full, sampling_time_test)

    split_time = train_split_index * sampling_time
    plot_full_trajectory("train_full", u_train_full, y_train_full, sampling_time, plots_path / "train_trajectory.png", split_time=split_time)
    plot_full_trajectory("test_full", u_test_full, y_test_full, sampling_time_test, plots_path / "test_trajectory.png")

    config_pars_general["n_states"] = int(validation_sequence.y0.shape[-1])
    save_config(config_pars_general)
    save_metadata(config_pars_general, validation_sequence, test_sequence)

    print("Prepared EMPS (single 80/20 split, continuous-horizon validation)")
    print(f"  train_split_index={train_split_index} (samples 0-{train_split_index-1} used for training windows)")
    print(f"  validation_sequence: shape_u={tuple(validation_sequence.u.shape)} warmup={validation_sequence.warmup} "
          f"(scored samples: {validation_sequence.num_samples - validation_sequence.warmup})")
    print(f"  test_sequence: warmup={test_sequence.warmup} shape_u={tuple(test_sequence.u.shape)}")
    print(f"Saved cached datasets in {base_path}")


if __name__ == "__main__":
    main()