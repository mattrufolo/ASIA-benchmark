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

# WienerHammerBenchMark (Schoukens, Suykens & Ljung, 2009): a real
# electronic Wiener-Hammerstein system -- LTI filter G1 (3rd order
# Chebyshev, 4.4kHz cutoff) -> static nonlinearity (diode circuit) -> LTI
# filter G2 (3rd order inverse Chebyshev, transmission zero at 5kHz).
# Officially downloadable, with a genuine train/test split -- unlike
# BoucWen, no self-generated data is needed here. See
# WienerHammerBenchMark_description.md for the full writeup.

input_names = ["voltage_in"]
output_names = ["voltage_out"]

config_pars_general = {
    "benchmark_name": "WienerHammerBenchMark",
    "device": "cpu",
    "seed": 42,
    # The official test warmup is exactly 50 samples. Using the same value
    # as history_window means the official warmup window is entirely spent
    # building the initial-condition vector, so warmup_test = 0.
    "history_window": 50,
    "benchmark_test_initialization_window_length": 50,
    "warmup_test": 0,  # recomputed in main()/loaders from the benchmark value
    # Leave-one-fold-out CV over the single 100000-sample train recording
    # (~1.95s at 51.2kHz): these are two slices of ONE continuous
    # recording (like CED), so contiguous folds are used, matching the
    # EMPS/CED project convention.
    "num_folds": 5,
    "window_min_length": 2000,
    "extend_windows_to_range_end": False,
    "crops_per_step": 1,
    "validation_metric": "rmse",
    "base_path": "./cached_data",
    "raw_data_path": "./data",
    "checkpoint_path": "./checkpoints",
    "plots_path": "./plots",
    "log_dir": "./logs",
    "eval_every": 25,
    "search_time_budget_seconds": 3600.0,
    "n_inputs": 1,
    "n_outputs": 1,
    "n_states": 100,  # recomputed below
}


@dataclass
class WHSequence:
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

    def clone(self) -> "WHSequence":
        return WHSequence(
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
    def fit(cls, sequences: list[WHSequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")
        u_all = torch.cat([s.u.reshape(-1, s.u.shape[-1]) for s in sequences], dim=0)
        y_all = torch.cat([s.y.reshape(-1, s.y.shape[-1]) for s in sequences], dim=0)
        return cls(
            u_mean=u_all.mean(dim=0, keepdim=True), u_std=u_all.std(dim=0, keepdim=True, unbiased=False),
            y_mean=y_all.mean(dim=0, keepdim=True), y_std=y_all.std(dim=0, keepdim=True, unbiased=False),
            history_window=history_window, n_inputs=int(sequences[0].u.shape[-1]), n_outputs=int(sequences[0].y.shape[-1]), eps=eps,
        )

    def normalize_sequence(self, sequence: WHSequence) -> WHSequence:
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
    train_tuple, test_tuple = nonlinear_benchmarks.WienerHammerBenchMark(
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


def make_sequence(name, u_full, y_full, sampling_time, start_index, stop_index, history_window, warmup=0) -> WHSequence:
    y0_vector = build_initial_condition(u_full, y_full, start_index, history_window)
    return WHSequence(
        name=name, u=torch.from_numpy(u_full[start_index:stop_index]).unsqueeze(0),
        y=torch.from_numpy(y_full[start_index:stop_index]).unsqueeze(0),
        y0=torch.from_numpy(y0_vector).view(1, 1, -1), sampling_time=sampling_time,
        warmup=warmup, start_index=start_index, stop_index=stop_index,
    )


def split_train_into_folds(u_full, y_full, sampling_time, num_folds, history_window) -> dict[str, WHSequence]:
    usable_start = history_window
    usable_length = len(u_full) - history_window
    fold_sizes = np.full(num_folds, usable_length // num_folds, dtype=int)
    fold_sizes[: usable_length % num_folds] += 1

    sequences: dict[str, WHSequence] = {}
    current_start = usable_start
    for fold_index, fold_size in enumerate(fold_sizes, start=1):
        current_stop = current_start + int(fold_size)
        name = f"fold_{fold_index}"
        sequences[name] = make_sequence(name, u_full, y_full, sampling_time, current_start, current_stop, history_window, warmup=0)
        current_start = current_stop
    return sequences


def load_raw_train_arrays() -> tuple[np.ndarray, np.ndarray, float]:
    path = RAW_DATA_DIR / "train_full_raw.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `prepare.py` first.")
    data = np.load(path)
    return np.asarray(data["u"], dtype=np.float32), np.asarray(data["y"], dtype=np.float32), float(data["sampling_time"])


def sample_training_window(u_full, y_full, sampling_time, history_window, valid_ranges, min_window_length, rng, extend_to_range_end=False) -> WHSequence:
    range_lengths = np.array([stop - start for start, stop in valid_ranges], dtype=np.float64)
    range_lengths = np.clip(range_lengths, a_min=0, a_max=None)
    if range_lengths.sum() <= 0:
        raise ValueError("No valid ranges with positive length were provided.")
    probabilities = range_lengths / range_lengths.sum()
    range_index = int(rng.choice(len(valid_ranges), p=probabilities))
    range_start, range_stop = valid_ranges[range_index]

    lo = max(range_start, history_window)
    max_start = range_stop - min_window_length
    if max_start < lo:
        start_index = lo
        window_length = max(1, range_stop - lo)
    elif extend_to_range_end:
        start_index = int(rng.integers(lo, max_start + 1))
        window_length = range_stop - start_index
    else:
        start_index = int(rng.integers(lo, max_start + 1))
        max_window_length = range_stop - start_index
        window_length = int(rng.integers(min_window_length, max_window_length + 1))
    stop = start_index + window_length
    return make_sequence(f"window_{start_index}_{stop}", u_full, y_full, sampling_time, start_index, stop, history_window, warmup=0)


def save_sequence(path: Path, sequence: WHSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, name=np.asarray(sequence.name), u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32), y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time), warmup=np.int64(sequence.warmup),
        start_index=np.int64(sequence.start_index), stop_index=np.int64(sequence.stop_index),
    )


def load_sequence(path: Path) -> WHSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")
    data = np.load(path, allow_pickle=True)
    name_value = data["name"]
    name = name_value.item() if np.asarray(name_value).shape == () else str(name_value)
    return WHSequence(
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


def load_train_sequences(base_path=None) -> dict[str, WHSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_dir = root / "train_sequences"
    sequence_paths = sorted(train_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached training sequences under {train_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def _finalize_config(config: dict, reference_sequence: WHSequence) -> dict:
    config["n_inputs"] = int(reference_sequence.u.shape[-1])
    config["n_outputs"] = int(reference_sequence.y.shape[-1])
    config["n_states"] = int(reference_sequence.y0.shape[-1])
    benchmark_window_length = int(config.get("benchmark_test_initialization_window_length", config_pars_general["benchmark_test_initialization_window_length"]))
    config["benchmark_test_initialization_window_length"] = benchmark_window_length
    config["warmup_test"] = effective_test_warmup(int(config["history_window"]), benchmark_window_length)
    return config


def load_datasets_and_config(base_path=None) -> tuple[dict[str, WHSequence], WHSequence, dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    test_sequence = load_sequence(root / "dataset_test.npz")
    config = load_config(root)
    reference_train = next(iter(train_sequences.values()))
    config = _finalize_config(config, reference_train)
    for sequence in train_sequences.values():
        sequence.warmup = 0
    test_sequence.warmup = int(config["warmup_test"])
    return train_sequences, test_sequence, config


def load_train_sequences_and_config(base_path=None) -> tuple[dict[str, WHSequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    config = load_config(root)
    reference_train = next(iter(train_sequences.values()))
    config = _finalize_config(config, reference_train)
    for sequence in train_sequences.values():
        sequence.warmup = 0
    return train_sequences, config


def plot_full_trajectory(name, u, y, sampling_time, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(len(u), dtype=np.float32) * sampling_time
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_axis, u[:, 0], linewidth=0.5, color="#006d77")
    axes[0].set_ylabel(input_names[0] + " [V]")
    axes[0].set_title(f"{name} input trajectory")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(time_axis, y[:, 0], linewidth=0.5, color="#bc6c25")
    axes[1].set_ylabel(output_names[0] + " [V]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title(f"{name} output trajectory")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_metadata(config, train_sequences, test_sequence) -> None:
    base_path = Path(config["base_path"])
    metadata = {
        "benchmark_name": config["benchmark_name"], "sampling_time": test_sequence.sampling_time,
        "history_window": config["history_window"], "num_folds": config["num_folds"],
        "train_sequences": [{"name": s.name, "num_samples": s.num_samples} for s in train_sequences.values()],
        "test_sequence": {"name": test_sequence.name, "num_samples": test_sequence.num_samples, "warmup": test_sequence.warmup},
        "notes": [
            "train and test are two slices of ONE continuous recording (like CED), "
            "so contiguous folds of the single train recording are used for "
            "leave-one-fold-out CV -- see WienerHammerBenchMark_description.md Section 5.",
            "Report units: u, y both in Volts. The official submission template "
            "reports RMSE in millivolts (RMSE_V * 1000).",
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

    train_sequences_dir = base_path / "train_sequences"
    if train_sequences_dir.exists():
        shutil.rmtree(train_sequences_dir)

    train_raw, test_raw = load_official_splits(force_download=False)
    u_train_full, y_train_full, sampling_time = benchmark_to_arrays(train_raw.atleast_2d())
    u_test_full, y_test_full, sampling_time_test = benchmark_to_arrays(test_raw.atleast_2d())

    history_window = int(config_pars_general["history_window"])
    benchmark_window_length = int(getattr(test_raw, "state_initialization_window_length", 50))
    config_pars_general["benchmark_test_initialization_window_length"] = benchmark_window_length
    config_pars_general["warmup_test"] = effective_test_warmup(history_window, benchmark_window_length)

    train_sequences = split_train_into_folds(u_train_full, y_train_full, sampling_time, config_pars_general["num_folds"], history_window)
    test_sequence = make_sequence(
        "test", u_test_full, y_test_full, sampling_time_test, history_window, len(u_test_full),
        history_window, warmup=config_pars_general["warmup_test"],
    )

    for name, sequence in train_sequences.items():
        save_sequence(train_sequences_dir / f"{name}.npz", sequence)
    save_sequence(base_path / "dataset_test.npz", test_sequence)
    save_raw_arrays(raw_data_path / "train_full_raw.npz", u_train_full, y_train_full, sampling_time)
    save_raw_arrays(raw_data_path / "test_full_raw.npz", u_test_full, y_test_full, sampling_time_test)

    plot_full_trajectory("train_full", u_train_full, y_train_full, sampling_time, plots_path / "train_trajectory.png")
    plot_full_trajectory("test_full", u_test_full, y_test_full, sampling_time_test, plots_path / "test_trajectory.png")

    config_pars_general["n_states"] = int(next(iter(train_sequences.values())).y0.shape[-1])
    save_config(config_pars_general)
    save_metadata(config_pars_general, train_sequences, test_sequence)

    print(f"Prepared {config_pars_general['benchmark_name']}")
    for name, sequence in sorted(train_sequences.items()):
        print(f"  {name}: shape_u={tuple(sequence.u.shape)} shape_y={tuple(sequence.y.shape)}")
    print(f"  test: warmup={test_sequence.warmup} shape_u={tuple(test_sequence.u.shape)} shape_y={tuple(test_sequence.y.shape)}")
    print(f"Saved cached datasets in {base_path}")


if __name__ == "__main__":
    main()