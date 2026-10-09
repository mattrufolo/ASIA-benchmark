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


SCRIPT_DIR = Path(__file__).resolve().parent

# The official nonlinear_benchmarks.FineSteeringMirror() loader is not
# yet available for this benchmark (confirmed by the user directly, as
# of this session) -- this project instead reads the same underlying
# data directly from the fsm-benchmark-data repo's own data/ folder
# (github.com/merijnfloren/fsm-benchmark-data), matching the pattern
# already used elsewhere in this collection (e.g. BoucWen, F16) when a
# benchmark's package loader isn't reliably available.
#
# Place the repo's data/ folder here (or point RAW_DATA_LOCAL_DIR at it):
#   <this project>/fsm-benchmark-data/data/u_100mV_train.npy, etc.
RAW_DATA_LOCAL_DIR = SCRIPT_DIR / "fsm-benchmark-data" / "data"

# 3 amplitude levels (100/200/300 mV RMS), each stored as
# u_{level}_{split}.npy / y_{level}_{split}.npy, shape (N, {nu,ny}, R, P)
# -- verified directly against the actual files this session: N=8192,
# R=6 (train) / R=3 (test), P=2, and the u RMS values match the paper's
# stated 100/200/300 mV levels exactly (98.9/197.9/296.9 mV measured).
#
# fs=6400Hz is hardcoded here (confirmed directly from the paper --
# Floren et al., ISMA-USD2024, Section 3.1) since it is NOT embedded as
# metadata in these files, unlike when loading through the official
# package (which exposes it via .sampling_time directly).
#
# history_window: the official package would normally provide this via
# .state_initialization_window_length -- not available here since we're
# reading local files directly. Set to a reasoned-but-not-authoritative
# default (see the value below); if you later gain access to the
# official loader's own value, prefer that over this one.
SAMPLING_FREQUENCY_HZ = 6400.0
DEFAULT_HISTORY_WINDOW = 100

# This project concatenates the P=2 periods within each realization into
# one continuous 16384-sample sequence (the excitation is genuinely
# periodic/continuous across the period boundary -- see
# FSM_description.md Section 2), giving 6 training sequences and 3
# official test sequences per amplitude level.
N_PERIODS = 2
N_REALIZATIONS_TRAIN = 6
N_REALIZATIONS_TEST = 3
AMPLITUDE_LEVELS_MV = [100, 200, 300]

input_names = ["actuator_1_voltage", "actuator_2_voltage", "actuator_3_voltage"]
output_names = ["displacement_1", "displacement_2", "displacement_3"]

config_pars_general = {
    "benchmark_name": "FSM",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    # Set from DEFAULT_HISTORY_WINDOW in main() -- see the note above;
    # this is a reasoned default, not an authoritative package-provided
    # value (the official loader isn't available for this benchmark yet).
    "history_window": DEFAULT_HISTORY_WINDOW,

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
    "n_inputs": 3,
    "n_outputs": 3,
    "n_states": 600,  # recomputed below (2 * history_window * (n_inputs+n_outputs))
}


@dataclass
class FSMSequence:
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

    def clone(self) -> "FSMSequence":
        return FSMSequence(
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
    def fit(cls, sequences: list[FSMSequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")
        u_all = torch.cat([s.u.reshape(-1, s.u.shape[-1]) for s in sequences], dim=0)
        y_all = torch.cat([s.y.reshape(-1, s.y.shape[-1]) for s in sequences], dim=0)
        return cls(
            u_mean=u_all.mean(dim=0, keepdim=True), u_std=u_all.std(dim=0, keepdim=True, unbiased=False),
            y_mean=y_all.mean(dim=0, keepdim=True), y_std=y_all.std(dim=0, keepdim=True, unbiased=False),
            history_window=history_window, n_inputs=int(sequences[0].u.shape[-1]), n_outputs=int(sequences[0].y.shape[-1]), eps=eps,
        )

    def normalize_sequence(self, sequence: FSMSequence) -> FSMSequence:
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


def _sequences_from_level_tensor(u_tensor, y_tensor, sampling_time, level_name, num_realizations) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """u_tensor/y_tensor: (N, n_channels, R, P). Concatenates the P
    periods within each realization into one continuous sequence (see
    FSM_description.md Section 2), returning a list of
    (name, u_seq (T,3), y_seq (T,3)) tuples, one per realization."""
    n_samples, n_channels_u, r_dim, p_dim = u_tensor.shape
    sequences = []
    for r in range(min(num_realizations, r_dim)):
        u_periods = [u_tensor[:, :, r, p] for p in range(p_dim)]  # each (N, n_channels)
        y_periods = [y_tensor[:, :, r, p] for p in range(p_dim)]
        u_seq = np.concatenate(u_periods, axis=0)  # (N*P, n_channels)
        y_seq = np.concatenate(y_periods, axis=0)
        name = f"{level_name}_r{r + 1:02d}"
        sequences.append((name, u_seq.astype(np.float32), y_seq.astype(np.float32)))
    return sequences


def build_initial_condition(u_full, y_full, start_index, history_window) -> np.ndarray:
    if start_index < history_window:
        raise ValueError("Not enough past samples to build the initial-condition vector.")
    u_history = np.stack([u_full[start_index - lag] for lag in range(1, history_window + 1)], axis=0)
    y_history = np.stack([y_full[start_index - lag] for lag in range(1, history_window + 1)], axis=0)
    return np.concatenate([u_history.reshape(-1), y_history.reshape(-1)], axis=0).astype(np.float32)


def make_sequence(name, u_full, y_full, sampling_time, start_index, stop_index, history_window,
                   warmup=0, fold="") -> FSMSequence:
    y0_vector = build_initial_condition(u_full, y_full, start_index, history_window)
    return FSMSequence(
        name=name, u=torch.from_numpy(u_full[start_index:stop_index]).view(1, -1, u_full.shape[-1]),
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


def sample_training_window(u_full, y_full, sampling_time, history_window, valid_ranges, min_window_length, rng, extend_to_range_end=False) -> FSMSequence:
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


def save_sequence(path: Path, sequence: FSMSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, name=np.asarray(sequence.name), u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32), y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time), warmup=np.int64(sequence.warmup), fold=np.asarray(sequence.fold),
    )


def load_sequence(path: Path) -> FSMSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")
    with np.load(path, allow_pickle=True) as data:
        def _scalar_str(value):
            arr = np.asarray(value)
            return arr.item() if arr.shape == () else str(value)
        return FSMSequence(
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


def load_train_sequences(base_path=None) -> dict[str, FSMSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_dir = root / "train_sequences"
    sequence_paths = sorted(train_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached training sequences under {train_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_test_sequences(base_path=None) -> dict[str, FSMSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    test_dir = root / "test_sequences"
    sequence_paths = sorted(test_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached test sequences under {test_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def _finalize_config(config: dict, reference_sequence: FSMSequence) -> dict:
    config["n_inputs"] = int(reference_sequence.u.shape[-1])
    config["n_outputs"] = int(reference_sequence.y.shape[-1])
    config["n_states"] = int(reference_sequence.y0.shape[-1])
    return config


def load_train_sequences_and_config(base_path=None) -> tuple[dict[str, FSMSequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    config = load_config(root)
    reference_train = next(iter(train_sequences.values()))
    config = _finalize_config(config, reference_train)
    for sequence in train_sequences.values():
        sequence.warmup = 0
    return train_sequences, config


def load_datasets_and_config(base_path=None) -> tuple[dict[str, FSMSequence], dict[str, FSMSequence], dict]:
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
    fig, axes = plt.subplots(6, 1, figsize=(11, 13), sharex=True)
    colors_u = ["#006d77", "#118ab2", "#073b4c"]
    colors_y = ["#bc6c25", "#6a4c93", "#e76f51"]
    for i in range(3):
        axes[i].plot(time_axis, u[:, i], linewidth=0.4, color=colors_u[i])
        axes[i].set_ylabel(f"{input_names[i]}\n[V]"); axes[i].grid(alpha=0.25)
    axes[0].set_title(f"{name} input")
    for i in range(3):
        axes[i + 3].plot(time_axis, y[:, i], linewidth=0.4, color=colors_y[i])
        axes[i + 3].set_ylabel(f"{output_names[i]}\n[m]"); axes[i + 3].grid(alpha=0.25)
    axes[3].set_title(f"{name} output")
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout(); fig.savefig(plot_path, dpi=140); plt.close(fig)


def load_level_from_local_files(level_mv: int, split: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Reads u_{level}mV_{split}.npy / y_{level}mV_{split}.npy directly
    from RAW_DATA_LOCAL_DIR -- (N, {nu,ny}, R, P) tensors, matching the
    exact shape/convention the official package would have returned
    (verified directly against these actual files: N=8192, R=6/3, P=2,
    u RMS matches the paper's stated 100/200/300mV levels)."""
    u_path = RAW_DATA_LOCAL_DIR / f"u_{level_mv}mV_{split}.npy"
    y_path = RAW_DATA_LOCAL_DIR / f"y_{level_mv}mV_{split}.npy"
    for path in (u_path, y_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file '{path}' not found. Copy the fsm-benchmark-data repo's 'data/' "
                f"folder (github.com/merijnfloren/fsm-benchmark-data) into this project's own "
                f"directory, so that '{path}' exists (see RAW_DATA_LOCAL_DIR at the top of this file)."
            )
    u = np.asarray(np.load(u_path), dtype=np.float32)
    y = np.asarray(np.load(y_path), dtype=np.float32)
    return u, y, 1.0 / SAMPLING_FREQUENCY_HZ


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

    trains = [load_level_from_local_files(level_mv, "train") for level_mv in AMPLITUDE_LEVELS_MV]
    tests = [load_level_from_local_files(level_mv, "test") for level_mv in AMPLITUDE_LEVELS_MV]

    history_window = int(config_pars_general["history_window"])  # local-file default -- see note near DEFAULT_HISTORY_WINDOW above

    train_sequences: dict[str, FSMSequence] = {}
    for level_idx, dataset in enumerate(trains):
        u_tensor, y_tensor, sampling_time = dataset
        level_name = f"level{level_idx + 1:02d}"
        level_sequences = _sequences_from_level_tensor(u_tensor, y_tensor, sampling_time, level_name, N_REALIZATIONS_TRAIN)
        for seq_name, u_seq, y_seq in level_sequences:
            sequence = make_sequence(seq_name, u_seq, y_seq, sampling_time, history_window, len(u_seq), history_window, warmup=0, fold=seq_name)
            train_sequences[seq_name] = sequence
            save_sequence(train_dir / f"{seq_name}.npz", sequence)
            save_raw_arrays(raw_data_path / f"{seq_name}_raw.npz", u_seq, y_seq, sampling_time)
        first_name, first_u, first_y = level_sequences[0]
        plot_full_trajectory(first_name, first_u, first_y, sampling_time, plots_path / f"{first_name}_trajectory.png")

    test_sequences: dict[str, FSMSequence] = {}
    for level_idx, dataset in enumerate(tests):
        u_tensor, y_tensor, sampling_time = dataset
        level_name = f"level{level_idx + 1:02d}"
        for seq_name, u_seq, y_seq in _sequences_from_level_tensor(u_tensor, y_tensor, sampling_time, level_name, N_REALIZATIONS_TEST):
            test_name = f"test_{seq_name}"
            sequence = make_sequence(test_name, u_seq, y_seq, sampling_time, history_window, len(u_seq), history_window, warmup=0)
            test_sequences[test_name] = sequence
            save_sequence(test_dir / f"{test_name}.npz", sequence)
            save_raw_arrays(raw_data_path / f"{test_name}_raw.npz", u_seq, y_seq, sampling_time)

    config_pars_general["n_states"] = int(next(iter(train_sequences.values())).y0.shape[-1])
    save_config(config_pars_general)

    print("Prepared CubeSpec Fine Steering Mirror (FSM) benchmark (via local files, fsm-benchmark-data/data/)")
    print(f"  fs = {SAMPLING_FREQUENCY_HZ} Hz (hardcoded, confirmed from the paper), history_window = {history_window} "
          f"samples -- a reasoned DEFAULT, not an authoritative package-provided value (the official "
          f"nonlinear_benchmarks loader isn't available for this benchmark yet)")
    print(f"  {len(train_sequences)} train_val sequences (6-fold leave-one-realization-out CV, 3 per fold, one per amplitude level):")
    for name, seq in sorted(train_sequences.items()):
        print(f"    {name}: {seq.num_samples} samples")
    print(f"  {len(test_sequences)} OFFICIAL test sequences (monitoring only):")
    for name, seq in sorted(test_sequences.items()):
        print(f"    {name}: {seq.num_samples} samples")
    print(f"Saved cached datasets in {base_path}")


if __name__ == "__main__":
    main()