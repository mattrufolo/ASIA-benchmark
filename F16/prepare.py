from __future__ import annotations

import pickle
import random
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
        "Missing dependency `nonlinear_benchmarks`. Install it first (`pip install nonlinear_benchmarks`)."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent

# nonlinear_benchmarks.F16() downloads and caches the official data itself
# on first call -- no manual data placement needed (unlike benchmarks in
# this collection whose package loader isn't reliably available, where a
# local copy of the raw files is required instead).
#
# train_val: 8 Input_output_data objects (4 FullMSine + 4 SineSw "odd
# level" estimation datasets, lengths 73728 / ~108k, confirmed directly
# against the package's own README this session).
# test: 6 Input_output_data objects (the paired "even level" official
# test datasets, same lengths).
#
# The package does not label which of the 8/6 datasets is which specific
# excitation level -- each Input_output_data object exposes .u, .y,
# .sampling_time, .state_initialization_window_length, but not a level
# number. This project distinguishes FullMSine vs SineSw datasets by
# LENGTH (73728 samples = FullMSine, everything else = SineSw -- verified
# directly against the actual data), which is reliable, but does NOT
# claim to know the exact excitation level (1/3/5/7) within each type --
# labels below are `{type}_train_{index}`, not `{type}_L{level}`, to
# avoid asserting something not actually confirmed.
FULLMSINE_LENGTH = 73728


def classify_signal_type(length: int) -> str:
    return "FullMSine" if length == FULLMSINE_LENGTH else "SineSw"


input_names = ["force"]
output_names = ["acceleration_excitation", "acceleration_wing", "acceleration_payload"]

config_pars_general = {
    "benchmark_name": "F16_GVT",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "fs": 400.0,
    # Overwritten in main() with the OFFICIAL
    # state_initialization_window_length reported by the package itself
    # (tests[0].state_initialization_window_length), rather than a
    # hand-picked value -- this value is what actually appears here once
    # prepare.py has been run at least once; see the printed/cached
    # config for the real number.
    "history_window": 200,
    "warmup_test": 0,
    "window_min_length": 1000,
    "extend_windows_to_range_end": False,
    "crops_per_step": 4,
    "validation_metric": "rmse",
    "base_path": "./cached_data",
    "raw_data_path": "./data",
    "checkpoint_path": "./checkpoints",
    "plots_path": "./plots",
    "log_dir": "./logs",
    "eval_every": 25,
    "search_time_budget_seconds": 1800.0,
    "n_inputs": 1,
    "n_outputs": 3,
    "n_states": 800,  # recomputed below (2 * history_window * (n_inputs+n_outputs))
}


@dataclass
class F16Sequence:
    name: str
    u: torch.Tensor
    y: torch.Tensor
    y0: torch.Tensor
    sampling_time: float
    warmup: int = 0
    fold: str = ""

    @property
    def num_samples(self) -> int:
        return int(self.u.shape[1])

    def clone(self) -> "F16Sequence":
        return F16Sequence(
            name=self.name, u=self.u.clone(), y=self.y.clone(), y0=self.y0.clone(),
            sampling_time=self.sampling_time, warmup=self.warmup, fold=self.fold,
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
    def fit(cls, sequences: list[F16Sequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")
        u_all = torch.cat([s.u.reshape(-1, s.u.shape[-1]) for s in sequences], dim=0)
        y_all = torch.cat([s.y.reshape(-1, s.y.shape[-1]) for s in sequences], dim=0)
        return cls(
            u_mean=u_all.mean(dim=0, keepdim=True), u_std=u_all.std(dim=0, keepdim=True, unbiased=False),
            y_mean=y_all.mean(dim=0, keepdim=True), y_std=y_all.std(dim=0, keepdim=True, unbiased=False),
            history_window=history_window, n_inputs=int(sequences[0].u.shape[-1]), n_outputs=int(sequences[0].y.shape[-1]), eps=eps,
        )

    def normalize_sequence(self, sequence: F16Sequence) -> F16Sequence:
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

    def normalize_y_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        std = self.y_std.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        return (values - mean) / std

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


def _extract_u_y_fs(dataset) -> tuple[np.ndarray, np.ndarray, float]:
    """Unpacks ONE Input_output_data object from nonlinear_benchmarks.F16()
    into plain numpy arrays. u -> (T,) or (T,1) depending on the package's
    own convention; normalized here to (T,) for u (n_inputs=1) and (T,3)
    for y (n_outputs=3), matching this project's own array conventions."""
    u, y = dataset  # Input_output_data objects unpack as (u, y), matching every other benchmark's own usage pattern
    u = np.asarray(u, dtype=np.float32).reshape(-1)  # (T,) -- single input (Force)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        # Should not happen for F16 (3 outputs), but handle defensively
        # in case the package's own array convention differs from what's
        # been confirmed for the other, single-output benchmarks.
        y = y.reshape(-1, 1)
    elif y.shape[0] == 3 and y.shape[1] != 3:
        # Defensive transpose in case the package returns (n_outputs, T)
        # rather than (T, n_outputs) for this multi-output benchmark.
        y = y.T
    fs = float(getattr(dataset, "sampling_time", None) or (1.0 / 400.0))
    sampling_time = fs if fs < 1.0 else 1.0 / fs  # normalize whichever convention was returned
    return u, y, sampling_time


def build_initial_condition(u_full, y_full, start_index, history_window) -> np.ndarray:
    if start_index < history_window:
        raise ValueError("Not enough past samples to build the initial-condition vector.")
    u_history = np.stack([u_full[start_index - lag] for lag in range(1, history_window + 1)], axis=0)
    y_history = np.stack([y_full[start_index - lag] for lag in range(1, history_window + 1)], axis=0)
    return np.concatenate([u_history.reshape(-1), y_history.reshape(-1)], axis=0).astype(np.float32)


def make_sequence(name, u_full, y_full, sampling_time, start_index, stop_index, history_window,
                   warmup=0, fold="") -> F16Sequence:
    y0_vector = build_initial_condition(u_full, y_full, start_index, history_window)
    return F16Sequence(
        name=name, u=torch.from_numpy(u_full[start_index:stop_index]).view(1, -1, 1),
        y=torch.from_numpy(y_full[start_index:stop_index]).view(1, -1, y_full.shape[-1]),
        y0=torch.from_numpy(y0_vector).view(1, 1, -1), sampling_time=sampling_time,
        warmup=warmup, fold=fold,
    )


def load_raw_sequence_array(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    path = Path(config_pars_general["raw_data_path"]) / f"{name}_raw.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `prepare.py` first.")
    with np.load(path) as data:
        return np.asarray(data["u"], dtype=np.float32), np.asarray(data["y"], dtype=np.float32), float(data["sampling_time"])


def sample_training_window(u_full, y_full, sampling_time, history_window, valid_ranges, min_window_length, rng, extend_to_range_end=False) -> F16Sequence:
    """Draws ONE freshly, randomly-positioned/randomly-sized crop from
    valid_ranges every training step -- these sequences are 70k-117k
    samples long, far too long to unroll fully through a per-timestep
    recurrent model every step."""
    range_index = int(rng.integers(0, len(valid_ranges)))
    lo_range, hi_range = valid_ranges[range_index]
    lo = max(lo_range, history_window)
    hi = hi_range
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


def save_sequence(path: Path, sequence: F16Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, name=np.asarray(sequence.name), u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32), y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time), warmup=np.int64(sequence.warmup), fold=np.asarray(sequence.fold),
    )


def load_sequence(path: Path) -> F16Sequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")
    with np.load(path, allow_pickle=True) as data:
        def _scalar_str(value):
            arr = np.asarray(value)
            return arr.item() if arr.shape == () else str(value)
        return F16Sequence(
            name=str(_scalar_str(data["name"])), u=torch.from_numpy(np.asarray(data["u"], dtype=np.float32)),
            y=torch.from_numpy(np.asarray(data["y"], dtype=np.float32)), y0=torch.from_numpy(np.asarray(data["y0"], dtype=np.float32)),
            sampling_time=float(data["sampling_time"]), warmup=int(data["warmup"]),
            fold=str(_scalar_str(data.get("fold", np.asarray("")))),
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


def load_train_sequences(base_path=None) -> dict[str, F16Sequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_dir = root / "train_sequences"
    sequence_paths = sorted(train_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached training sequences under {train_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_test_sequences(base_path=None) -> dict[str, F16Sequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    test_dir = root / "test_sequences"
    sequence_paths = sorted(test_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached test sequences under {test_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def _finalize_config(config: dict, reference_sequence: F16Sequence) -> dict:
    config["n_inputs"] = int(reference_sequence.u.shape[-1])
    config["n_outputs"] = int(reference_sequence.y.shape[-1])
    config["n_states"] = int(reference_sequence.y0.shape[-1])
    return config


def load_train_sequences_and_config(base_path=None) -> tuple[dict[str, F16Sequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    config = load_config(root)
    reference_train = next(iter(train_sequences.values()))
    config = _finalize_config(config, reference_train)
    for sequence in train_sequences.values():
        sequence.warmup = 0
    return train_sequences, config


def load_datasets_and_config(base_path=None) -> tuple[dict[str, F16Sequence], dict[str, F16Sequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    test_sequences = load_test_sequences(root)
    config = load_config(root)
    reference_train = next(iter(train_sequences.values()))
    config = _finalize_config(config, reference_train)
    for sequence in train_sequences.values():
        sequence.warmup = 0
    for sequence in test_sequences.values():
        sequence.warmup = int(config["warmup_test"])
    return train_sequences, test_sequences, config


def plot_full_trajectory(name, u, y, sampling_time, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(len(u), dtype=np.float32) * sampling_time
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(time_axis, u, linewidth=0.4, color="#006d77")
    axes[0].set_ylabel("force [N]"); axes[0].set_title(f"{name} input"); axes[0].grid(alpha=0.25)
    colors = ["#bc6c25", "#6a4c93", "#e76f51"]
    for i, oname in enumerate(output_names):
        axes[i + 1].plot(time_axis, y[:], linewidth=0.4, color=colors[i])
        axes[i + 1].set_ylabel(f"{oname}\n[m/s^2]")
        axes[i + 1].grid(alpha=0.25)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(plot_path, dpi=150); plt.close(fig)


def main() -> None:
    set_global_seed(config_pars_general["seed"])
    base_path = Path(config_pars_general["base_path"])
    raw_data_path = Path(config_pars_general["raw_data_path"])
    plots_path = Path(config_pars_general["plots_path"])

    for p in (base_path, raw_data_path, plots_path):
        p.mkdir(parents=True, exist_ok=True)

    train_dir = base_path / "train_sequences"
    test_dir = base_path / "test_sequences"
    for directory in (train_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for f in directory.glob("*.npz"):
            f.unlink()

    # Downloads/caches automatically on first call -- no manual data
    # placement needed, unlike benchmarks in this collection whose
    # package loader isn't reliably available.
    trains, tests = nonlinear_benchmarks.F16()
    if not isinstance(trains, (list, tuple)):
        trains = [trains]
    if not isinstance(tests, (list, tuple)):
        tests = [tests]

    # Official state-initialization window length, taken directly from
    # the package itself rather than hand-picked -- overwrites the
    # config_pars_general default above.
    history_window = int(getattr(tests[0], "state_initialization_window_length", config_pars_general["history_window"]))
    config_pars_general["history_window"] = history_window

    type_counters = {"FullMSine": 0, "SineSw": 0}
    train_sequences: dict[str, F16Sequence] = {}
    for dataset in trains:
        u, y, sampling_time = _extract_u_y_fs(dataset)
        signal_type = classify_signal_type(len(u))
        type_counters[signal_type] += 1
        fold_name = f"{signal_type}_train_{type_counters[signal_type]:02d}"
        sequence = make_sequence(fold_name, u, y, sampling_time, history_window, len(u), history_window, warmup=0, fold=fold_name)
        train_sequences[fold_name] = sequence
        save_sequence(train_dir / f"{fold_name}.npz", sequence)
        save_raw_arrays(raw_data_path / f"{fold_name}_raw.npz", u, y, sampling_time)
        plot_full_trajectory(fold_name, u, y, sampling_time, plots_path / f"{fold_name}_trajectory.png")

    type_counters_test = {"FullMSine": 0, "SineSw": 0}
    test_sequences: dict[str, F16Sequence] = {}
    for dataset in tests:
        u, y, sampling_time = _extract_u_y_fs(dataset)
        signal_type = classify_signal_type(len(u))
        type_counters_test[signal_type] += 1
        test_name = f"test_{signal_type}_{type_counters_test[signal_type]:02d}"
        sequence = make_sequence(test_name, u, y, sampling_time, history_window, len(u), history_window, warmup=0)
        test_sequences[test_name] = sequence
        save_sequence(test_dir / f"{test_name}.npz", sequence)
        save_raw_arrays(raw_data_path / f"{test_name}_raw.npz", u, y, sampling_time)
        plot_full_trajectory(test_name, u, y, sampling_time, plots_path / f"{test_name}_trajectory.png")

    config_pars_general["n_states"] = int(next(iter(train_sequences.values())).y0.shape[-1])
    save_config(config_pars_general)

    print("Prepared F16 GVT benchmark (via nonlinear_benchmarks.F16(), Force -> 3 accelerations)")
    print(f"  fs = {config_pars_general['fs']} Hz, history_window = {history_window} samples "
          f"({history_window/config_pars_general['fs']:.3f}s) -- taken from the package's own "
          f"state_initialization_window_length")
    print(f"  {len(train_sequences)} train_val sequences (leave-one-out CV folds):")
    for name, seq in sorted(train_sequences.items()):
        print(f"    {name}: {seq.num_samples} samples ({seq.num_samples/config_pars_general['fs']:.1f}s)")
    print(f"  {len(test_sequences)} OFFICIAL test sequences (monitoring only):")
    for name, seq in sorted(test_sequences.items()):
        print(f"    {name}: {seq.num_samples} samples ({seq.num_samples/config_pars_general['fs']:.1f}s)")
    print(f"Saved cached datasets in {base_path}")
    print("NOTE: dataset names indicate signal TYPE (verified by length) but not the specific")
    print("excitation LEVEL (1/3/5/7) within that type -- the package itself doesn't label this.")


if __name__ == "__main__":
    main()