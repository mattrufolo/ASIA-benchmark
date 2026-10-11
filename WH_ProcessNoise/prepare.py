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
    import scipy.io
except ImportError as exc:
    raise SystemExit("Missing dependency `scipy`. Install it first (`pip install scipy`).") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = SCRIPT_DIR / "data"
# This benchmark's data is NOT available through the `nonlinear_benchmarks`
# package (confirmed not present/working for this project) -- instead,
# place the OFFICIAL zip's extracted contents directly here, i.e. copy the
# `WienerHammersteinFiles/` folder from the official download
# (data.4tu.nl, article 12952124) into this project's own directory, so
# the layout looks like:
#   WienerHammersteinProcessNoise/
#     WienerHammersteinFiles/
#       README.txt
#       WH_EstimationExample.mat
#       WH_TestDataset.mat
#       WH_MultisineFadeOut.mat
#       ... (other measurement-campaign files, not required by default)
# Only `WH_EstimationExample.mat` and `WH_TestDataset.mat` are read by
# this project's default protocol; the folder name/location below must
# match exactly where those two files actually are.
RAW_MAT_DIR = SCRIPT_DIR / "WienerHammersteinFiles"

# Wiener-Hammerstein benchmark WITH PROCESS NOISE (Schoukens & Noel,
# 2016/2017): R(s) -> [+ process noise e_x(t)] -> f(x) [diode-resistor
# static nonlinearity] -> S(s) -> y. The process noise enters INSIDE the
# system, before the nonlinearity -- dominant noise source, genuinely
# different from the plain WienerHammerBenchMark project in this
# collection (which has only small measurement noise). Two much smaller,
# separate measurement noise sources are added on the input/output
# channels. fs=78125 Hz. Output dynamics have a transmission zero (same
# identification challenge the plain WienerHammerBenchMark's S(s) has).
#
# Confirmed directly from the official zip's own README.txt AND by
# loading the actual .mat files (not just secondary sources):
#   WH_EstimationExample.mat : 10 realizations x 8192 samples (2 periods
#                               of 4096 pts), WITH process noise. Chosen
#                               as this project's DEFAULT estimation set
#                               -- literally named for this purpose, and
#                               gives much richer folding options than
#                               the alternative below.
#   WH_MultisineFadeOut.mat  : 2 realizations x 8192 samples, fading to
#                               zero at the end, WITH process noise. The
#                               dataset most third-party papers actually
#                               used (a literature-comparable ALTERNATIVE,
#                               not the default here -- see
#                               WienerHammersteinProcessNoise_description.md
#                               Section 4 for why 10 realizations was
#                               chosen as the default instead).
#   WH_TestDataset.mat       : 16384 samples x 2 columns = [multisine
#                               test, sine-sweep test]. NOISELESS
#                               (process noise switched off for testing)
#                               -- "not to be used during model
#                               estimation" per the official README.
#                               Reported SEPARATELY, never combined.
# Several additional measurement-campaign files also exist in the
# official zip (combined multisine+sweep+zero, pure sine sweep, discrete
# sine sweep, triangle-superposed inputs x2, zero input) -- not used by
# this project's default protocol (to stay comparable with the
# literature), but their raw arrays are cached too, for anyone who wants
# to experiment with using additional (noisy) training data as an
# explicit search candidate.

input_names = ["input_voltage"]
output_names = ["output_voltage"]

config_pars_general = {
    "benchmark_name": "WienerHammersteinProcessNoise",
    "device": "cpu",
    "seed": 42,
    "sampling_rate_hz": 78125.0,
    "num_estimation_realizations": 10,
    "num_folds": 5,  # 5 folds of 2 realizations each, out of the 10 available
    "history_window": 20,  # matches a third-party paper's own IC convention for this exact benchmark
    "warmup_test": 0,  # test sequences have no natural warmup beyond history_window; see loaders
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
    "search_time_budget_seconds": 600.0,
    "n_inputs": 1,
    "n_outputs": 1,
    "n_states": 40,  # recomputed below
}


@dataclass
class WHPSequence:
    name: str
    u: torch.Tensor
    y: torch.Tensor
    y0: torch.Tensor
    sampling_time: float
    warmup: int = 0
    fold: str = ""
    realization_index: int = -1

    @property
    def num_samples(self) -> int:
        return int(self.u.shape[1])

    def clone(self) -> "WHPSequence":
        return WHPSequence(
            name=self.name, u=self.u.clone(), y=self.y.clone(), y0=self.y0.clone(),
            sampling_time=self.sampling_time, warmup=self.warmup, fold=self.fold,
            realization_index=self.realization_index,
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
    def fit(cls, sequences: list[WHPSequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")
        u_all = torch.cat([s.u.reshape(-1, s.u.shape[-1]) for s in sequences], dim=0)
        y_all = torch.cat([s.y.reshape(-1, s.y.shape[-1]) for s in sequences], dim=0)
        return cls(
            u_mean=u_all.mean(dim=0, keepdim=True), u_std=u_all.std(dim=0, keepdim=True, unbiased=False),
            y_mean=y_all.mean(dim=0, keepdim=True), y_std=y_all.std(dim=0, keepdim=True, unbiased=False),
            history_window=history_window, n_inputs=int(sequences[0].u.shape[-1]), n_outputs=int(sequences[0].y.shape[-1]), eps=eps,
        )

    def normalize_sequence(self, sequence: WHPSequence) -> WHPSequence:
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


def load_official_mat_files():
    """Reads the two .mat files this project needs DIRECTLY from the
    local `WienerHammersteinFiles/` folder (see RAW_MAT_DIR's own comment
    above for the exact expected layout) -- no `nonlinear_benchmarks`
    package dependency and no network access, since this benchmark's
    data isn't reliably available through that package. Each .mat file's
    `dataMeas` struct has fields (r, u, y, fs): `r` is the reference/
    programmed input (before measurement noise), `u`/`y` are the actual
    measured signals used here; `r` is intentionally unused."""

    def _load(stem: str):
        path = RAW_MAT_DIR / f"{stem}.mat"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected file '{path}' not found. Copy the official zip's extracted "
                f"'WienerHammersteinFiles/' folder (from data.4tu.nl, article 12952124) "
                f"into this project's own directory, so that '{path}' exists. "
                f"Files currently found in {RAW_MAT_DIR}: "
                f"{sorted(p.name for p in RAW_MAT_DIR.glob('*.mat')) if RAW_MAT_DIR.exists() else '(directory does not exist)'}"
            )
        out = scipy.io.loadmat(str(path))
        r, u, y, fs = out["dataMeas"][0, 0]
        return np.asarray(u, dtype=np.float32), np.asarray(y, dtype=np.float32), float(fs.item())

    estimation_u, estimation_y, fs_est = _load("WH_EstimationExample")
    test_u, test_y, fs_test = _load("WH_TestDataset")
    return (estimation_u, estimation_y, fs_est), (test_u, test_y, fs_test)


def build_initial_condition(u_full, y_full, start_index, history_window) -> np.ndarray:
    if start_index < history_window:
        raise ValueError("Not enough past samples to build the initial-condition vector.")
    u_history = [u_full[start_index - lag] for lag in range(1, history_window + 1)]
    y_history = [y_full[start_index - lag] for lag in range(1, history_window + 1)]
    ic_vector = np.concatenate([np.concatenate(u_history, axis=0), np.concatenate(y_history, axis=0)], axis=0)
    return ic_vector.astype(np.float32)


def make_sequence(name, u_full, y_full, sampling_time, start_index, stop_index, history_window,
                   warmup=0, fold="", realization_index=-1) -> WHPSequence:
    y0_vector = build_initial_condition(u_full, y_full, start_index, history_window)
    return WHPSequence(
        name=name, u=torch.from_numpy(u_full[start_index:stop_index]).unsqueeze(0),
        y=torch.from_numpy(y_full[start_index:stop_index]).unsqueeze(0),
        y0=torch.from_numpy(y0_vector).view(1, 1, -1), sampling_time=sampling_time,
        warmup=warmup, fold=fold, realization_index=realization_index,
    )


def load_raw_sequence_array(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    path = RAW_DATA_DIR / f"{name}_raw.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `prepare.py` first.")
    with np.load(path) as data:
        return np.asarray(data["u"], dtype=np.float32), np.asarray(data["y"], dtype=np.float32), float(data["sampling_time"])


def sample_training_window(u_full, y_full, sampling_time, history_window, min_window_length, rng, extend_to_range_end=False) -> WHPSequence:
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


def save_sequence(path: Path, sequence: WHPSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, name=np.asarray(sequence.name), u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32), y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time), warmup=np.int64(sequence.warmup),
        fold=np.asarray(sequence.fold), realization_index=np.int64(sequence.realization_index),
    )


def load_sequence(path: Path) -> WHPSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")
    with np.load(path, allow_pickle=True) as data:
        def _scalar_str(value):
            arr = np.asarray(value)
            return arr.item() if arr.shape == () else str(value)
        return WHPSequence(
            name=str(_scalar_str(data["name"])), u=torch.from_numpy(np.asarray(data["u"], dtype=np.float32)),
            y=torch.from_numpy(np.asarray(data["y"], dtype=np.float32)), y0=torch.from_numpy(np.asarray(data["y0"], dtype=np.float32)),
            sampling_time=float(data["sampling_time"]), warmup=int(data["warmup"]),
            fold=str(_scalar_str(data.get("fold", np.asarray("")))), realization_index=int(data.get("realization_index", -1)),
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


def load_train_sequences(base_path=None) -> dict[str, WHPSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_dir = root / "train_sequences"
    sequence_paths = sorted(train_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached training sequences under {train_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_test_sequences(base_path=None) -> dict[str, WHPSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    test_dir = root / "test_sequences"
    sequence_paths = sorted(test_dir.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached test sequences under {test_dir}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def group_by_fold(sequences: dict[str, WHPSequence]) -> dict[str, list[WHPSequence]]:
    groups: dict[str, list[WHPSequence]] = {}
    for sequence in sequences.values():
        groups.setdefault(sequence.fold, []).append(sequence)
    return groups


def _finalize_config(config: dict, reference_sequence: WHPSequence) -> dict:
    config["n_inputs"] = int(reference_sequence.u.shape[-1])
    config["n_outputs"] = int(reference_sequence.y.shape[-1])
    config["n_states"] = int(reference_sequence.y0.shape[-1])
    return config


def load_train_sequences_and_config(base_path=None) -> tuple[dict[str, WHPSequence], dict]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    train_sequences = load_train_sequences(root)
    config = load_config(root)
    reference_train = next(iter(train_sequences.values()))
    config = _finalize_config(config, reference_train)
    for sequence in train_sequences.values():
        sequence.warmup = 0
    return train_sequences, config


def load_datasets_and_config(base_path=None) -> tuple[dict[str, WHPSequence], dict[str, WHPSequence], dict]:
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
    metadata = {
        "benchmark_name": config["benchmark_name"], "history_window": config["history_window"],
        "num_folds": config["num_folds"], "num_estimation_realizations": config["num_estimation_realizations"],
        "num_train_sequences": len(train_sequences), "num_test_sequences": len(test_sequences),
        "notes": [
            "Default estimation set: WH_EstimationExample.mat, 10 independent "
            "realizations x 8192 samples each, WITH dominant process noise "
            "(entering the system BEFORE the static nonlinearity -- see "
            "WienerHammersteinProcessNoise_description.md). Grouped into 5 "
            "folds of 2 realizations each for leave-one-fold-out CV.",
            "Test: WH_TestDataset.mat, 2 columns = [multisine test, "
            "sine-sweep test], 16384 samples each, NOISELESS (process noise "
            "switched off for testing) -- reported SEPARATELY, never "
            "combined, per the official README's own instruction not to "
            "use this data during estimation.",
            "The official zip's OTHER measurement-campaign files (a "
            "literature-standard 2-realization alternative estimation set, "
            "plus several additional recordings) are cached separately "
            "under data/ but NOT used by this project's default protocol "
            "-- see program.md for how to try them as explicit experiments.",
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

    (estimation_u, estimation_y, fs_est), (test_u, test_y, fs_test) = load_official_mat_files()
    sampling_time_est = 1.0 / fs_est
    sampling_time_test = 1.0 / fs_test

    history_window = int(config_pars_general["history_window"])
    num_realizations = int(config_pars_general["num_estimation_realizations"])
    num_folds = int(config_pars_general["num_folds"])
    realizations_per_fold = num_realizations // num_folds

    train_sequences: dict[str, WHPSequence] = {}
    for realization in range(num_realizations):
        fold_index = realization // realizations_per_fold
        fold_name = f"fold_{fold_index + 1}"
        u_full = estimation_u[:, realization:realization + 1]
        y_full = estimation_y[:, realization:realization + 1]
        name = f"{fold_name}_realization{realization:02d}"
        sequence = make_sequence(
            name, u_full, y_full, sampling_time_est, history_window, len(u_full), history_window,
            warmup=0, fold=fold_name, realization_index=realization,
        )
        train_sequences[name] = sequence
        save_sequence(train_sequences_dir / f"{name}.npz", sequence)
        save_raw_arrays(raw_data_path / f"{name}_raw.npz", u_full, y_full, sampling_time_est)
        if realization == 0:
            plot_full_trajectory(name, u_full, y_full, sampling_time_est, plots_path / "train_example_trajectory.png")

    test_names = ["test_multisine", "test_sinesweep"]
    test_sequences: dict[str, WHPSequence] = {}
    for column, name in enumerate(test_names):
        u_full = test_u[:, column:column + 1]
        y_full = test_y[:, column:column + 1]
        sequence = make_sequence(
            name, u_full, y_full, sampling_time_test, history_window, len(u_full), history_window,
            warmup=0,
        )
        test_sequences[name] = sequence
        save_sequence(test_sequences_dir / f"{name}.npz", sequence)
        save_raw_arrays(raw_data_path / f"{name}_raw.npz", u_full, y_full, sampling_time_test)
        plot_full_trajectory(name, u_full, y_full, sampling_time_test, plots_path / f"{name}_trajectory.png")

    config_pars_general["n_states"] = int(next(iter(train_sequences.values())).y0.shape[-1])
    save_config(config_pars_general)
    save_metadata(config_pars_general, train_sequences, test_sequences)

    print(f"Prepared {config_pars_general['benchmark_name']}")
    fold_counts: dict[str, int] = {}
    for sequence in train_sequences.values():
        fold_counts[sequence.fold] = fold_counts.get(sequence.fold, 0) + 1
    for fold_name, count in sorted(fold_counts.items()):
        print(f"  {fold_name}: {count} realizations (shape_u per realization = {tuple(next(iter(train_sequences.values())).u.shape)})")
    for name, sequence in sorted(test_sequences.items()):
        print(f"  {name}: shape_u={tuple(sequence.u.shape)} shape_y={tuple(sequence.y.shape)}")
    print(f"Saved cached datasets in {base_path}")


if __name__ == "__main__":
    main()