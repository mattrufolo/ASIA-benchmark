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

# The official nonlinear_benchmarks.CED() loader only uses the *uniformly
# distributed* input data set (DATAUNIF.MAT in the technical report), which
# contains two 500-sample realizations recorded at Ts=0.02 s:
#   - "low input amplitude"  (u11 / z11, PRBS switching -1.5V/+2.5V)
#   - "high input amplitude" (u12 / z12, PRBS switching -1.0V/+3.0V)
# For each realization, samples [0:400] are the official train/val split and
# samples [400:500] are the official held-out test split, with a mandatory
# state_initialization_window_length of 10 samples on the test side.
# (The three PRBS-only sequences u1/u2/u3 described in the report are *not*
# part of the official CED() benchmark split and are intentionally not used
# here, to stay compatible with the official leaderboard protocol.)

input_names = ["combined_motor_voltage"]
output_names = ["pulley_speed_ticks_per_s"]

REALIZATIONS = ("low_amplitude", "high_amplitude")

# --- Procedure used in this project (see program.md and CED_description.md) ---
#
# 1. SEARCH:  train on [history_window:search_train_stop] of BOTH regimes
#             jointly, validate on [search_train_stop:400] of BOTH regimes
#             jointly (a single held-out window per regime, adjacent to the
#             official test window, instead of interchangeable CV folds).
#             Architecture/hyperparameter decisions are made ONLY on this
#             search-validation RMSE.
# 2. REFIT:   once a winning config is chosen, retrain that SAME config on
#             the FULL [history_window:400] of both regimes (no held-out
#             split; the [search_train_stop:400] window is now training
#             data and must never again be used to pick between models).
# 3. TEST:    evaluate the refit model once on the official [400:500] test
#             split. This number is for reporting only and must not be used
#             to go back and reconsider the choice made in step 1.
#
# The plot of the raw trajectories (see plots/*_train_trajectory.png) shows
# the official test window is a calmer, lower-amplitude continuation of the
# recording, qualitatively different from the busy middle section. Splitting
# search-validation off the *end* of the train window (rather than an
# interior fold) puts the validation data in the same regime as the actual
# test data, which is the point of this design.

config_pars_general = {
    "benchmark_name": "Coupled_Electric_Drives",
    "device": "cpu",
    "seed": 42,
    # Absolute sample index (within each 400-sample train realization) where
    # the search-train / search-val boundary sits. search-train covers
    # [history_window:search_train_stop], search-val covers
    # [search_train_stop:search_val_stop].
    "search_train_stop": 350,
    "search_val_stop": 400,
    # The official test warmup is exactly 10 samples. Using the same value
    # as history_window means the official warmup window is entirely spent
    # building the initial-condition vector, and warmup_test becomes 0 (the
    # full remaining 90 test samples per realization are scored).
    "history_window": 10,
    # Minimum length of the randomly-sampled training windows used by
    # train.py/refit.py (see `sample_training_window`). A new random window
    # (random start index AND random length, both >= this floor) is drawn
    # every training step, instead of always unrolling one fixed full-length
    # sequence -- this gives the initial-condition network many different
    # history-window examples to learn to generalize from, instead of just
    # one per regime.
    "window_min_length": 60,
    "benchmark_test_initialization_window_length": 10,
    "warmup_test": 0,  # recomputed in main()/loaders from the benchmark value
    "validation_metric": "rmse",
    "base_path": "./cached_data",
    "raw_data_path": "./data",
    "checkpoint_path": "./checkpoints",
    "plots_path": "./plots",
    "log_dir": "./logs",
    "eval_every": 25,
    "search_time_budget_seconds": 3000.0,
    # REFIT (step 2) has no held-out validation data, so it cannot early-stop
    # on val RMSE. Instead it watches its own full-window training RMSE
    # (a stable quantity, unlike the noisy per-step random-window loss) and
    # stops once that plateaus, bounded by this time budget as a safety net.
    "refit_time_budget_seconds": 600.0,
    "refit_patience": 20,
    "n_inputs": 1,
    "n_outputs": 1,
    "n_states": 20,  # 2 * history_window * (n_inputs + n_outputs) / 2, recomputed below
}


@dataclass
class CEDSequence:
    name: str
    u: torch.Tensor
    y: torch.Tensor
    y0: torch.Tensor
    sampling_time: float
    realization: str = ""
    warmup: int = 0
    start_index: int = 0
    stop_index: int = 0

    @property
    def num_samples(self) -> int:
        return int(self.u.shape[1])

    def clone(self) -> "CEDSequence":
        return CEDSequence(
            name=self.name,
            u=self.u.clone(),
            y=self.y.clone(),
            y0=self.y0.clone(),
            sampling_time=self.sampling_time,
            realization=self.realization,
            warmup=self.warmup,
            start_index=self.start_index,
            stop_index=self.stop_index,
        )


class Normalizer:
    def __init__(
        self,
        u_mean: torch.Tensor,
        u_std: torch.Tensor,
        y_mean: torch.Tensor,
        y_std: torch.Tensor,
        history_window: int,
        n_inputs: int,
        n_outputs: int,
        eps: float = 1e-6,
    ) -> None:
        self.u_mean = u_mean.float()
        self.u_std = torch.clamp(u_std.float(), min=eps)
        self.y_mean = y_mean.float()
        self.y_std = torch.clamp(y_std.float(), min=eps)
        self.history_window = int(history_window)
        self.n_inputs = int(n_inputs)
        self.n_outputs = int(n_outputs)
        self.eps = eps

    @classmethod
    def fit(cls, sequences: list[CEDSequence], history_window: int, eps: float = 1e-6) -> "Normalizer":
        if not sequences:
            raise ValueError("At least one training sequence is required to fit the normalizer.")

        u_all = torch.cat([sequence.u.reshape(-1, sequence.u.shape[-1]) for sequence in sequences], dim=0)
        y_all = torch.cat([sequence.y.reshape(-1, sequence.y.shape[-1]) for sequence in sequences], dim=0)

        u_mean = u_all.mean(dim=0, keepdim=True)
        u_std = u_all.std(dim=0, keepdim=True, unbiased=False)
        y_mean = y_all.mean(dim=0, keepdim=True)
        y_std = y_all.std(dim=0, keepdim=True, unbiased=False)

        return cls(
            u_mean=u_mean,
            u_std=u_std,
            y_mean=y_mean,
            y_std=y_std,
            history_window=history_window,
            n_inputs=int(sequences[0].u.shape[-1]),
            n_outputs=int(sequences[0].y.shape[-1]),
            eps=eps,
        )

    def normalize_sequence(self, sequence: CEDSequence) -> CEDSequence:
        normalized = sequence.clone()
        normalized.u = (normalized.u - self.u_mean.unsqueeze(0)) / self.u_std.unsqueeze(0)
        normalized.y = (normalized.y - self.y_mean.unsqueeze(0)) / self.y_std.unsqueeze(0)

        u_hist_dim = self.history_window * self.n_inputs
        y_hist_dim = self.history_window * self.n_outputs

        u_mean_hist = self.u_mean.repeat(1, self.history_window)
        u_std_hist = self.u_std.repeat(1, self.history_window)
        y_mean_hist = self.y_mean.repeat(1, self.history_window)
        y_std_hist = self.y_std.repeat(1, self.history_window)

        ic_mean = torch.cat([u_mean_hist, y_mean_hist], dim=1).unsqueeze(0)
        ic_std = torch.cat([u_std_hist, y_std_hist], dim=1).unsqueeze(0)

        if sequence.y0.shape[-1] != u_hist_dim + y_hist_dim:
            raise ValueError("Initial-condition vector size is inconsistent with history_window.")

        normalized.y0 = (normalized.y0 - ic_mean) / ic_std
        return normalized

    def denormalize_y_tensor(self, values: torch.Tensor) -> torch.Tensor:
        mean = self.y_mean.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        std = self.y_std.to(device=values.device, dtype=values.dtype).unsqueeze(0)
        return values * std + mean

    def state_dict(self) -> dict[str, np.ndarray | float | int]:
        return {
            "u_mean": self.u_mean.detach().cpu().numpy(),
            "u_std": self.u_std.detach().cpu().numpy(),
            "y_mean": self.y_mean.detach().cpu().numpy(),
            "y_std": self.y_std.detach().cpu().numpy(),
            "history_window": int(self.history_window),
            "n_inputs": int(self.n_inputs),
            "n_outputs": int(self.n_outputs),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, np.ndarray | float | int]) -> "Normalizer":
        return cls(
            u_mean=torch.as_tensor(state["u_mean"], dtype=torch.float32),
            u_std=torch.as_tensor(state["u_std"], dtype=torch.float32),
            y_mean=torch.as_tensor(state["y_mean"], dtype=torch.float32),
            y_std=torch.as_tensor(state["y_std"], dtype=torch.float32),
            history_window=int(state["history_window"]),
            n_inputs=int(state["n_inputs"]),
            n_outputs=int(state["n_outputs"]),
            eps=float(state.get("eps", 1e-6)),
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
    """Returns (train_tuple, test_tuple), each a 2-tuple of Input_output_data:
    index 0 = low input amplitude realization, index 1 = high input amplitude.
    """
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    train_tuple, test_tuple = nonlinear_benchmarks.CED(
        atleast_2d=True,
        always_return_tuples_of_datasets=True,
        dir_placement=str(DOWNLOAD_DIR),
        force_download=force_download,
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


def build_initial_condition(
    u_full: np.ndarray,
    y_full: np.ndarray,
    start_index: int,
    history_window: int,
) -> np.ndarray:
    if start_index < history_window:
        raise ValueError("Not enough past samples to build the initial-condition vector.")

    u_history = [u_full[start_index - lag] for lag in range(1, history_window + 1)]
    y_history = [y_full[start_index - lag] for lag in range(1, history_window + 1)]

    ic_vector = np.concatenate(
        [
            np.concatenate(u_history, axis=0),
            np.concatenate(y_history, axis=0),
        ],
        axis=0,
    )
    return ic_vector.astype(np.float32)


def effective_test_warmup(history_window: int, benchmark_window_length: int) -> int:
    """Translate the benchmark warmup to the cropped test sequence used by this repo."""
    return max(0, int(benchmark_window_length) - int(history_window))


def make_sequence(
    name: str,
    u_full: np.ndarray,
    y_full: np.ndarray,
    sampling_time: float,
    start_index: int,
    stop_index: int,
    history_window: int,
    realization: str = "",
    warmup: int = 0,
) -> CEDSequence:
    y0_vector = build_initial_condition(
        u_full=u_full,
        y_full=y_full,
        start_index=start_index,
        history_window=history_window,
    )
    return CEDSequence(
        name=name,
        u=torch.from_numpy(u_full[start_index:stop_index]).unsqueeze(0),
        y=torch.from_numpy(y_full[start_index:stop_index]).unsqueeze(0),
        y0=torch.from_numpy(y0_vector).view(1, 1, -1),
        sampling_time=sampling_time,
        realization=realization,
        warmup=warmup,
        start_index=start_index,
        stop_index=stop_index,
    )


def save_sequence(path: Path, sequence: CEDSequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        name=np.asarray(sequence.name),
        u=sequence.u.detach().cpu().numpy().astype(np.float32),
        y=sequence.y.detach().cpu().numpy().astype(np.float32),
        y0=sequence.y0.detach().cpu().numpy().astype(np.float32),
        sampling_time=np.float32(sequence.sampling_time),
        realization=np.asarray(sequence.realization),
        warmup=np.int64(sequence.warmup),
        start_index=np.int64(sequence.start_index),
        stop_index=np.int64(sequence.stop_index),
    )


def load_sequence(path: Path) -> CEDSequence:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached sequence: {path}. Run `prepare.py` first.")

    data = np.load(path, allow_pickle=True)

    def _scalar_str(value) -> str:
        return value.item() if np.asarray(value).shape == () else str(value)

    return CEDSequence(
        name=_scalar_str(data["name"]),
        u=torch.from_numpy(np.asarray(data["u"], dtype=np.float32)),
        y=torch.from_numpy(np.asarray(data["y"], dtype=np.float32)),
        y0=torch.from_numpy(np.asarray(data["y0"], dtype=np.float32)),
        sampling_time=float(data["sampling_time"]),
        realization=_scalar_str(data["realization"]) if "realization" in data else "",
        warmup=int(data["warmup"]),
        start_index=int(data.get("start_index", 0)),
        stop_index=int(data.get("stop_index", 0)),
    )


def save_raw_arrays(path: Path, u: np.ndarray, y: np.ndarray, sampling_time: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=u.astype(np.float32),
        y=y.astype(np.float32),
        sampling_time=np.float32(sampling_time),
    )


def load_raw_train_arrays(realization: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Loads the FULL [0:400] raw train arrays for one realization, saved by
    `main()`. Used to sample random training windows (see
    `sample_training_window`) instead of always unrolling one fixed,
    full-length sequence -- the fixed-window approach only ever gives the
    initial-condition network ONE example per regime to learn from, which
    generalizes poorly to the (differently-positioned) initial-condition
    window built at test time. See CED_description.md Section 5 / 6."""
    path = RAW_DATA_DIR / f"{realization}_train_raw.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run `prepare.py` first.")
    data = np.load(path)
    return np.asarray(data["u"], dtype=np.float32), np.asarray(data["y"], dtype=np.float32), float(data["sampling_time"])


def sample_training_window(
    u_full: np.ndarray,
    y_full: np.ndarray,
    sampling_time: float,
    realization: str,
    history_window: int,
    stop_index: int,
    min_window_length: int,
    rng: np.random.Generator,
) -> CEDSequence:
    """Samples ONE random contiguous window from `u_full`/`y_full`, with a
    random start index (>= history_window, so the IC vector is always built
    from real past data) and a random length between `min_window_length` and
    the maximum available before `stop_index`. `stop_index` must be
    <= search_train_stop during SEARCH (step 1) or <= len(u_full) during
    REFIT (step 2), so this never reaches into `search_val`/test data.
    """
    max_start = stop_index - min_window_length
    if max_start < history_window:
        raise ValueError(
            f"stop_index={stop_index} and min_window_length={min_window_length} leave no room "
            f"for a valid window after history_window={history_window}."
        )

    start_index = int(rng.integers(history_window, max_start + 1))
    max_window_length = stop_index - start_index
    window_length = int(rng.integers(min_window_length, max_window_length + 1))
    stop = start_index + window_length

    return make_sequence(
        name=f"{realization}_window_{start_index}_{stop}",
        u_full=u_full,
        y_full=y_full,
        sampling_time=sampling_time,
        start_index=start_index,
        stop_index=stop,
        history_window=history_window,
        realization=realization,
        warmup=0,
    )


def save_config(config: dict) -> None:
    base_path = Path(config["base_path"])
    base_path.mkdir(parents=True, exist_ok=True)
    with (base_path / "config_params_general.pkl").open("wb") as handle:
        pickle.dump(config, handle)


def load_config(base_path: str | Path | None = None) -> dict:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    path = root / "config_params_general.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached config: {path}. Run `prepare.py` first.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_sequences_dir(directory: Path) -> dict[str, CEDSequence]:
    sequence_paths = sorted(directory.glob("*.npz"))
    if not sequence_paths:
        raise FileNotFoundError(f"Missing cached sequences under {directory}. Run `prepare.py` first.")
    return {path.stem: load_sequence(path) for path in sequence_paths}


def load_search_train_sequences(base_path: str | Path | None = None) -> dict[str, CEDSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return _load_sequences_dir(root / "search_train_sequences")


def load_search_val_sequences(base_path: str | Path | None = None) -> dict[str, CEDSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return _load_sequences_dir(root / "search_val_sequences")


def load_refit_train_sequences(base_path: str | Path | None = None) -> dict[str, CEDSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return _load_sequences_dir(root / "refit_train_sequences")


def load_test_sequences(base_path: str | Path | None = None) -> dict[str, CEDSequence]:
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    return _load_sequences_dir(root / "test_sequences")


def _finalize_config(config: dict, reference_sequence: CEDSequence) -> dict:
    config["n_inputs"] = int(reference_sequence.u.shape[-1])
    config["n_outputs"] = int(reference_sequence.y.shape[-1])
    config["n_states"] = int(reference_sequence.y0.shape[-1])
    benchmark_window_length = int(
        config.get(
            "benchmark_test_initialization_window_length",
            config_pars_general["benchmark_test_initialization_window_length"],
        )
    )
    config["benchmark_test_initialization_window_length"] = benchmark_window_length
    config["warmup_test"] = effective_test_warmup(
        history_window=int(config["history_window"]),
        benchmark_window_length=benchmark_window_length,
    )
    return config


def load_search_datasets_and_config(
    base_path: str | Path | None = None,
) -> tuple[dict[str, CEDSequence], dict[str, CEDSequence], dict]:
    """For step 1 (SEARCH): architecture/hyperparameter decisions are made
    using only these two dicts (search-train, search-val)."""
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    search_train = load_search_train_sequences(root)
    search_val = load_search_val_sequences(root)
    config = load_config(root)

    reference_sequence = next(iter(search_train.values()))
    config = _finalize_config(config, reference_sequence)
    for sequence in search_train.values():
        sequence.warmup = 0
    for sequence in search_val.values():
        sequence.warmup = 0

    return search_train, search_val, config


def load_refit_datasets_and_config(
    base_path: str | Path | None = None,
) -> tuple[dict[str, CEDSequence], dict[str, CEDSequence], dict]:
    """For step 2 (REFIT) and step 3 (TEST): the full [history_window:400]
    training data (both regimes) and the official test sequences. The
    [search_train_stop:400] window is INSIDE refit_train here -- it must
    never again be used to choose between models once this function is used."""
    root = Path(config_pars_general["base_path"] if base_path is None else base_path)
    refit_train = load_refit_train_sequences(root)
    test_sequences = load_test_sequences(root)
    config = load_config(root)

    reference_sequence = next(iter(refit_train.values()))
    config = _finalize_config(config, reference_sequence)
    for sequence in refit_train.values():
        sequence.warmup = 0
    for sequence in test_sequences.values():
        sequence.warmup = int(config["warmup_test"])

    return refit_train, test_sequences, config


def plot_full_trajectory(
    name: str,
    u: np.ndarray,
    y: np.ndarray,
    sampling_time: float,
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(len(u), dtype=np.float32) * sampling_time

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_axis, u[:, 0], linewidth=1.2, color="#006d77")
    axes[0].set_ylabel(input_names[0])
    axes[0].set_title(f"{name} input trajectory")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, y[:, 0], linewidth=1.2, color="#bc6c25")
    axes[1].set_ylabel(output_names[0])
    axes[1].set_xlabel("time [s]")
    axes[1].set_title(f"{name} output trajectory")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_metadata(
    config: dict,
    search_train_sequences: dict[str, CEDSequence],
    search_val_sequences: dict[str, CEDSequence],
    refit_train_sequences: dict[str, CEDSequence],
    test_sequences: dict[str, CEDSequence],
) -> None:
    base_path = Path(config["base_path"])
    metadata = {
        "benchmark_name": config["benchmark_name"],
        "validation_metric": config["validation_metric"],
        "device": config["device"],
        "history_window": config["history_window"],
        "search_train_stop": config["search_train_stop"],
        "search_val_stop": config["search_val_stop"],
        "sampling_time": next(iter(test_sequences.values())).sampling_time,
        "initial_condition_dimension": int(next(iter(test_sequences.values())).y0.shape[-1]),
        "input_names": input_names,
        "output_names": output_names,
        "search_train_sequences": [
            {"name": s.name, "realization": s.realization, "start_index": s.start_index, "stop_index": s.stop_index, "num_samples": s.num_samples}
            for s in search_train_sequences.values()
        ],
        "search_val_sequences": [
            {"name": s.name, "realization": s.realization, "start_index": s.start_index, "stop_index": s.stop_index, "num_samples": s.num_samples}
            for s in search_val_sequences.values()
        ],
        "refit_train_sequences": [
            {"name": s.name, "realization": s.realization, "start_index": s.start_index, "stop_index": s.stop_index, "num_samples": s.num_samples}
            for s in refit_train_sequences.values()
        ],
        "test_sequences": [
            {"name": s.name, "realization": s.realization, "num_samples": s.num_samples, "warmup": s.warmup}
            for s in test_sequences.values()
        ],
        "notes": [
            "Only the official DATAUNIF realizations (low/high input amplitude) are used, "
            "matching nonlinear_benchmarks.CED().",
            "PROCEDURE (see program.md / CED_description.md for the full rationale):",
            "  1. SEARCH: train.py trains on search_train_sequences (both regimes, "
            "[history_window:search_train_stop]) and validates on search_val_sequences "
            "(both regimes, [search_train_stop:search_val_stop]). Architecture/hyperparameter "
            "decisions are made ONLY on this search-validation RMSE.",
            "  2. REFIT: once a config is chosen, refit.py retrains that SAME config on "
            "refit_train_sequences (both regimes, the FULL [history_window:400], no held-out "
            "split) for a fixed epoch count derived from the search run's best_epoch.",
            "  3. TEST: test.py evaluates the refit checkpoint once on test_sequences "
            "(official [400:500]). This number is for reporting only.",
            "Each sequence starts only after `history_window` past samples are available "
            "(true past data, never zero-padded); the initial-condition vector is "
            "[u(k-1)...u(k-H), y(k-1)...y(k-H)] with H=history_window.",
            "The official test split provides 2 sequences of 100 samples each (indices "
            "400:500 of each realization) with state_initialization_window_length=10; with "
            "history_window=10 this official warmup is fully consumed by the "
            "initial-condition vector, so warmup_test=0.",
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

    search_train_dir = base_path / "search_train_sequences"
    search_val_dir = base_path / "search_val_sequences"
    refit_train_dir = base_path / "refit_train_sequences"
    test_dir = base_path / "test_sequences"
    for directory in (search_train_dir, search_val_dir, refit_train_dir, test_dir):
        if directory.exists():
            shutil.rmtree(directory)

    train_tuple, test_tuple = load_official_splits(force_download=False)

    history_window = int(config_pars_general["history_window"])
    search_train_stop = int(config_pars_general["search_train_stop"])
    search_val_stop = int(config_pars_general["search_val_stop"])

    search_train_sequences: dict[str, CEDSequence] = {}
    search_val_sequences: dict[str, CEDSequence] = {}
    refit_train_sequences: dict[str, CEDSequence] = {}
    test_sequences: dict[str, CEDSequence] = {}

    for realization, train_raw, test_raw in zip(REALIZATIONS, train_tuple, test_tuple):
        u_train_full, y_train_full, ts_train = benchmark_to_arrays(train_raw.atleast_2d())
        u_test_full, y_test_full, ts_test = benchmark_to_arrays(test_raw.atleast_2d())

        if search_val_stop > len(u_train_full):
            raise ValueError(
                f"search_val_stop={search_val_stop} exceeds the {len(u_train_full)} available "
                f"train samples for realization {realization}."
            )

        benchmark_window_length = int(getattr(test_raw, "state_initialization_window_length", 0))
        config_pars_general["benchmark_test_initialization_window_length"] = benchmark_window_length
        warmup_test = effective_test_warmup(
            history_window=history_window,
            benchmark_window_length=benchmark_window_length,
        )
        config_pars_general["warmup_test"] = warmup_test

        # --- Step 1 data: SEARCH train / SEARCH val ---
        search_train_sequences[f"{realization}_search_train"] = make_sequence(
            name=f"{realization}_search_train",
            u_full=u_train_full,
            y_full=y_train_full,
            sampling_time=ts_train,
            start_index=history_window,
            stop_index=search_train_stop,
            history_window=history_window,
            realization=realization,
            warmup=0,
        )
        search_val_sequences[f"{realization}_search_val"] = make_sequence(
            name=f"{realization}_search_val",
            u_full=u_train_full,
            y_full=y_train_full,
            sampling_time=ts_train,
            start_index=search_train_stop,
            stop_index=search_val_stop,
            history_window=history_window,
            realization=realization,
            warmup=0,
        )

        # --- Step 2 data: REFIT train (full [history_window:400], no held-out split) ---
        refit_train_sequences[f"{realization}_refit_train"] = make_sequence(
            name=f"{realization}_refit_train",
            u_full=u_train_full,
            y_full=y_train_full,
            sampling_time=ts_train,
            start_index=history_window,
            stop_index=len(u_train_full),
            history_window=history_window,
            realization=realization,
            warmup=0,
        )

        # --- Step 3 data: official TEST ---
        test_sequences[f"test_{realization}"] = make_sequence(
            name=f"test_{realization}",
            u_full=u_test_full,
            y_full=y_test_full,
            sampling_time=ts_test,
            start_index=history_window,
            stop_index=len(u_test_full),
            history_window=history_window,
            realization=realization,
            warmup=warmup_test,
        )

        save_raw_arrays(raw_data_path / f"{realization}_train_raw.npz", u_train_full, y_train_full, ts_train)
        save_raw_arrays(raw_data_path / f"{realization}_test_raw.npz", u_test_full, y_test_full, ts_test)
        plot_full_trajectory(f"{realization}_train", u_train_full, y_train_full, ts_train, plots_path / f"{realization}_train_trajectory.png")
        plot_full_trajectory(f"{realization}_test", u_test_full, y_test_full, ts_test, plots_path / f"{realization}_test_trajectory.png")

    for name, sequence in search_train_sequences.items():
        save_sequence(search_train_dir / f"{name}.npz", sequence)
    for name, sequence in search_val_sequences.items():
        save_sequence(search_val_dir / f"{name}.npz", sequence)
    for name, sequence in refit_train_sequences.items():
        save_sequence(refit_train_dir / f"{name}.npz", sequence)
    for name, sequence in test_sequences.items():
        save_sequence(test_dir / f"{name}.npz", sequence)

    reference_sequence = next(iter(search_train_sequences.values()))
    config_pars_general["n_states"] = int(reference_sequence.y0.shape[-1])
    save_config(config_pars_general)
    save_metadata(
        config_pars_general,
        search_train_sequences,
        search_val_sequences,
        refit_train_sequences,
        test_sequences,
    )

    print(f"Prepared {config_pars_general['benchmark_name']}")
    print("Step 1 (SEARCH) -- train sequences:")
    for name, sequence in sorted(search_train_sequences.items()):
        print(f"  {name}: [{sequence.start_index}:{sequence.stop_index}] shape_u={tuple(sequence.u.shape)}")
    print("Step 1 (SEARCH) -- val sequences:")
    for name, sequence in sorted(search_val_sequences.items()):
        print(f"  {name}: [{sequence.start_index}:{sequence.stop_index}] shape_u={tuple(sequence.u.shape)}")
    print("Step 2 (REFIT) -- full train sequences:")
    for name, sequence in sorted(refit_train_sequences.items()):
        print(f"  {name}: [{sequence.start_index}:{sequence.stop_index}] shape_u={tuple(sequence.u.shape)}")
    print("Step 3 (TEST) -- official test sequences:")
    for name, sequence in sorted(test_sequences.items()):
        print(f"  {name}: warmup={sequence.warmup} shape_u={tuple(sequence.u.shape)}")
    print(f"Saved cached datasets in {base_path}")
    print(f"Saved trajectory plots in {plots_path}")


if __name__ == "__main__":
    main()