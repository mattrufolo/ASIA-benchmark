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

# Parallel Wiener-Hammerstein benchmark (Schoukens, Marconato, Pintelon,
# Vandersteen & Rolain, 2017): a real electronic 2-branch parallel
# Wiener-Hammerstein system -- each branch is front LTI (3rd order) ->
# static diode-resistor nonlinearity -> back LTI (3rd order), branch
# outputs summed. Officially downloadable, with a genuinely rich
# structure: random-phase multisine excitation at 5 different RMS levels
# (100mV-1V, linearly spaced), 20 independent phase realizations per
# level for training/estimation, and ONE held-out phase realization per
# level for the official test. See ParallelWH_description.md for the
# full writeup and the fold-design rationale.

input_names = ["voltage_in"]
output_names = ["voltage_out"]

config_pars_general = {
    "benchmark_name": "ParallelWH",
    "device": "cpu",
    "seed": 42,
    "num_phases": 20,
    "num_amplitudes": 5,
    # 5 folds by PHASE GROUP (4 phases each), each fold spanning ALL 5
    # amplitude levels -- NOT folds by amplitude. The official test task
    # is "generalize to an unseen phase, at amplitude levels already seen
    # in training" (all 5 RMS levels appear in both train and test), so
    # amplitude-based folds would test the wrong generalization axis
    # entirely (a lesson learned from the CED project's original fold
    # design mismatch). See ParallelWH_description.md Section 5.
    "num_folds": 5,
    "history_window": 50,
    "benchmark_test_initialization_window_length": 50,
    "warmup_test": 0,  # recomputed in main()/loaders from the benchmark value
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
    "search_time_budget_seconds": 600.0,
    "n_inputs": 1,
    "n_outputs": 1,
    "n_states": 100,  # recomputed below
}


@dataclass
class PWHSequence:
    name: str
    u: torch.Tensor
    y: torch.Tensor
    y0: torch.Tensor
    sampling_time: float
    warmup: int = 0
    fold: str = ""
    phase: int = -1
    amplitude_index: int = -1

    @property
    def num_samples(self) -> int:
        return int(self.u.shape[1])

    def clone(self) -> "PWHSequence":
        return PWHSequence(
            name=self.name, u=self.u.clone(), y=self.y.clone(), y0=self.y0.clone(),
            sampling_time=self.sampling_time, warmup=self.warmup, fold=self.fold,
            phase=self.phase, amplitude_index=self.amplitude_index,
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
    def fit(cls, sequences: list[PWHSequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")
        u_all = torch.cat([s.u.reshape(-1, s.u.shape[-1]) for s in sequences], dim=0)
        y_all = torch.cat([s.y.reshape(-1, s.y.shape[-1]) for s in sequences], dim=0)
        return cls(
            u_mean=u_all.mean(dim=0, keepdim=True), u_std=u_all.std(dim=0, keepdim=True, unbiased=False),
            y_mean=y_all.mean(dim=0, keepdim=True), y_std=y_all.std(dim=0, keepdim=True, unbiased=False),
            history_window=history_window, n_inputs=int(sequences[0].u.shape[-1]), n_outputs=int(sequences[0].y.shape[-1]), eps=eps,
        )

    def normalize_sequence(self, sequence: PWHSequence) -> PWHSequence:
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
    """Returns (train_datasets, test_datasets): train_datasets is a tuple
    of 100 datasets (20 phases x 5 amplitudes, named 'Est-phase-P-amp-A'),
    test_datasets is a tuple of 5 datasets (1 held-out phase per
    amplitude, named 'Val-amp-A')."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    train_tuple, test_tuple = nonlinear_benchmarks.ParWH(
        atleast_2d=True, always_return_tuples_of_datasets=True,
        dir_placement=str(DOWNLOAD_DIR), force_download=force_download,
    )
    return train_tuple, test_tuple


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


def make_sequence(name, u_full, y_full, sampling_time, start_index, stop_index, history_window,
                   warmup=0, fold="", phase=-1, amplitude_index=-1) -> PWHSequence:
    y0_vector = build_initial_condition(u_full, y_full, start_index, history_window)
    return PWHSequence(
        name=name, u=torch.from_numpy(u_full[start_index:stop_index]).unsqueeze(0),
        y=torch.from_numpy(y_full[start_index:stop_index]).unsqueeze(0),
        y0=torch.from_numpy(y0_vector).view(1, 1, -1), sampling_time=sampling_time,
        warmup=warmup, fold=fold, phase=phase, amplitude_index=amplitude_index,
    )


def load_raw_sequence_array(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Loads the full raw (u, y) array for one individual named sequence
    (e.g. 'Est-phase-03-amp-2'), saved by main()."""
    path = RAW_DATA_DIR / f"{name}_raw.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `prepare.py` first.")
    with np.load(path) as data:
        return np.asarray(data["u"], dtype=np.float32), np.asarray(data["y"], dtype=np.float32), float(data["sampling_time"])


def sample_training_window(u_full, y_full, sampling_time, history_window, min_window_length, rng, extend_to_range_end=False) -> PWHSequence:
    """Samples ONE random window from a SINGLE given sequence's full array
    (start index random, length random unless extend_to_range_end)."""
    n = len(u_full)
    lo = history_window
    max_start = n - min_window_length
    if max_start < lo:
        start_index = lo
        window_length = max(1, n - lo)
    elif extend_to_range_end:
        start_index = int(rng.integers(lo, max_start + 1))
        window_length = n - start_index
    else:
        start_index = int(rng.integers(lo, max_start + 1))
        max_window_length = n - start_index
        window_length = int(rng.integers(min_window_length, max_window_length + 1))
    stop = start_index + window_length
    return make_sequence(f"window_{start_index}_{stop}", u_full, y_full, sampling_time, start_index, stop, history_window, warmup=0)


def save_sequence(path: Path, sequence: PWHSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, name=np.asarray(sequence.name), u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32), y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time), warmup=np.int64(sequence.warmup),
        fold=np.asarray(sequence.fold), phase=np.int64(sequence.phase), amplitude_index=np.int64(sequence.amplitude_index),
    )


def load_sequence(path: Path) -> PWHSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")
    with np.load(path, allow_pickle=True) as data:
        def _scalar_str(value):
            arr = np.asarray(value)
            return arr.item() if arr.shape == () else str(value)
        return PWHSequence(
            name=str(_scalar_str(data["name"])), u=torch.from_numpy(np.asarray(data["u"], dtype=np.float32)),
            y=torch.from_numpy(np.asarray(data["y"], dtype=np.float32)), y0=torch.from_numpy(np.asarray(data["y0"], dtype=np.float32)),
            sampling_time=float(data["sampling_time"]), warmup=int(data["warmup"]),
            fold=str(_scalar_str(data.get("fold", np.asarray("")))), phase=int(data.get("phase", -1)),
            amplitude_index=int(data.get("amplitude_index", -1)),
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


def load_train_sequences(base_path=None) -> dict[str, PWHSequence]:
    """Returns ALL 100 individual training sequences, keyed by name. Fold
    membership is stored on each sequence's `.fold` attribute -- group by
    that to get per-fold lists."""
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_dir = root / "train_sequences"
    sequence_paths = sorted(train_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached training sequences under {train_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_test_sequences(base_path=None) -> dict[str, PWHSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    test_dir = root / "test_sequences"
    sequence_paths = sorted(test_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached test sequences under {test_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def group_by_fold(sequences: dict[str, PWHSequence]) -> dict[str, list[PWHSequence]]:
    groups: dict[str, list[PWHSequence]] = {}
    for sequence in sequences.values():
        groups.setdefault(sequence.fold, []).append(sequence)
    return groups


def _finalize_config(config: dict, reference_sequence: PWHSequence) -> dict:
    config["n_inputs"] = int(reference_sequence.u.shape[-1])
    config["n_outputs"] = int(reference_sequence.y.shape[-1])
    config["n_states"] = int(reference_sequence.y0.shape[-1])
    benchmark_window_length = int(config.get("benchmark_test_initialization_window_length", config_pars_general["benchmark_test_initialization_window_length"]))
    config["benchmark_test_initialization_window_length"] = benchmark_window_length
    config["warmup_test"] = effective_test_warmup(int(config["history_window"]), benchmark_window_length)
    return config


def load_train_sequences_and_config(base_path=None) -> tuple[dict[str, PWHSequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    config = load_config(root)
    reference_train = next(iter(train_sequences.values()))
    config = _finalize_config(config, reference_train)
    for sequence in train_sequences.values():
        sequence.warmup = 0
    return train_sequences, config


def load_datasets_and_config(base_path=None) -> tuple[dict[str, PWHSequence], dict[str, PWHSequence], dict]:
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


def save_metadata(config, train_sequences, test_sequences) -> None:
    base_path = Path(config["base_path"])
    any_seq = next(iter(train_sequences.values()))
    metadata = {
        "benchmark_name": config["benchmark_name"], "sampling_time": any_seq.sampling_time,
        "history_window": config["history_window"], "num_folds": config["num_folds"],
        "num_phases": config["num_phases"], "num_amplitudes": config["num_amplitudes"],
        "num_train_sequences": len(train_sequences), "num_test_sequences": len(test_sequences),
        "notes": [
            "100 training sequences (20 phases x 5 amplitude levels, 100mV-1V "
            "linearly spaced), grouped into 5 folds by PHASE GROUP (4 phases "
            "each), each fold spanning ALL 5 amplitude levels. This matches "
            "the official test task exactly (generalize to an unseen phase, "
            "at amplitude levels already seen in training) -- folding by "
            "amplitude instead would test the wrong generalization axis. See "
            "ParallelWH_description.md Section 5.",
            "5 official test sequences (Val-amp-0..4): one held-out phase "
            "realization per amplitude level, never used in any fold. "
            "Reported SEPARATELY per amplitude level in test.py, matching "
            "the paper's own Table 1 format.",
            "Each sequence is 2 concatenated periods (32768 samples) of a "
            "random-phase multisine, band [fs/N, 20kHz], measured at "
            "~78kHz (8x downsampled from the 625kHz generator clock).",
            "Report units: u, y in Volts. The paper reports validation "
            "error in millivolts (RMSE_V * 1000) per amplitude level.",
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
    test_sequences_dir = base_path / "test_sequences"
    for directory in (train_sequences_dir, test_sequences_dir):
        if directory.exists():
            shutil.rmtree(directory)

    train_datasets, test_datasets = load_official_splits(force_download=False)
    # train_datasets: 100 datasets named 'Est-phase-P-amp-A'; test_datasets:
    # 5 datasets named 'Val-amp-A'.

    history_window = int(config_pars_general["history_window"])
    num_phases = int(config_pars_general["num_phases"])
    num_amplitudes = int(config_pars_general["num_amplitudes"])
    num_folds = int(config_pars_general["num_folds"])
    phases_per_fold = num_phases // num_folds

    benchmark_window_length = int(getattr(test_datasets[0], "state_initialization_window_length", 50))
    config_pars_general["benchmark_test_initialization_window_length"] = benchmark_window_length
    config_pars_general["warmup_test"] = effective_test_warmup(history_window, benchmark_window_length)

    train_sequences: dict[str, PWHSequence] = {}
    for phase in range(num_phases):
        fold_index = phase // phases_per_fold
        fold_name = f"fold_{fold_index + 1}"
        for amp in range(num_amplitudes):
            dataset = train_datasets[phase * num_amplitudes + amp]
            u_full, y_full, sampling_time = benchmark_to_arrays(dataset.atleast_2d())
            name = f"{fold_name}_phase{phase:02d}_amp{amp}"
            sequence = make_sequence(
                name, u_full, y_full, sampling_time, history_window, len(u_full), history_window,
                warmup=0, fold=fold_name, phase=phase, amplitude_index=amp,
            )
            train_sequences[name] = sequence
            save_sequence(train_sequences_dir / f"{name}.npz", sequence)
            save_raw_arrays(raw_data_path / f"{name}_raw.npz", u_full, y_full, sampling_time)
            if phase == 0:
                plot_full_trajectory(name, u_full, y_full, sampling_time, plots_path / f"train_example_amp{amp}_trajectory.png")

    test_sequences: dict[str, PWHSequence] = {}
    for amp in range(num_amplitudes):
        dataset = test_datasets[amp]
        u_full, y_full, sampling_time = benchmark_to_arrays(dataset.atleast_2d())
        name = f"test_amp{amp}"
        sequence = make_sequence(
            name, u_full, y_full, sampling_time, history_window, len(u_full), history_window,
            warmup=config_pars_general["warmup_test"], amplitude_index=amp,
        )
        test_sequences[name] = sequence
        save_sequence(test_sequences_dir / f"{name}.npz", sequence)
        save_raw_arrays(raw_data_path / f"{name}_raw.npz", u_full, y_full, sampling_time)
        plot_full_trajectory(name, u_full, y_full, sampling_time, plots_path / f"{name}_trajectory.png")

    config_pars_general["n_states"] = int(next(iter(train_sequences.values())).y0.shape[-1])
    save_config(config_pars_general)
    save_metadata(config_pars_general, train_sequences, test_sequences)

    print(f"Prepared {config_pars_general['benchmark_name']}")
    fold_counts: dict[str, int] = {}
    for sequence in train_sequences.values():
        fold_counts[sequence.fold] = fold_counts.get(sequence.fold, 0) + 1
    for fold_name, count in sorted(fold_counts.items()):
        print(f"  {fold_name}: {count} sequences (shape_u per sequence = {tuple(next(iter(train_sequences.values())).u.shape)})")
    for name, sequence in sorted(test_sequences.items()):
        print(f"  {name}: warmup={sequence.warmup} shape_u={tuple(sequence.u.shape)} shape_y={tuple(sequence.y.shape)}")
    print(f"Saved cached datasets in {base_path}")


if __name__ == "__main__":
    main()