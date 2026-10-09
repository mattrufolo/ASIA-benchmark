from __future__ import annotations

import copy
import csv
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import prepare
from model import build_model_from_config


config_pars = {
    "lr": 8e-3,
    "max_epochs": 3000,
    "type": "PHYS_RECURRENT",
    "recurrent": "LSTM",
    "filter_length": 300,
    "n_hidden_states": 32,
    "n_hidden_states": 32,
    "hidden_sizes": [32],
    "activation": "Tanh",
    "num_layers": 1,
    "dropout_prob": 0.0,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 15,
    "direct_feedthrough": True,
    "use_output_feedback": False,  # set True to also support prediction-mode evaluation (see model.py)
    "extend_windows_to_range_end": False,
    "crops_per_step": 4,
    "fold_time_budget_seconds": None,  # None -> use prepare.py's shared default
    "device_override": None,  # None -> use prepare.py's auto-detected device (GPU if available); set "cpu" to force CPU
    "eval_every": 100,                # None -> use prepare.py's shared default
}


def append_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def initialize_fold_log(log_path: Path, fold_name: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{fold_name}\n")
        handle.write("epoch,elapsed_seconds,window_loss_norm_mse,train_rmse_norm,val_rmse_norm\n")


def compute_rmse(y_true, y_pred, warmup=0) -> float:
    return prepare.rmse(y_true[warmup:], y_pred[warmup:])


def build_model(model_config, general_config):
    return build_model_from_config(
        config_pars=model_config, n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"], n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])


def model_requires_raw_io(model) -> bool:
    """Models whose parameters are calibrated to real physical units
    (e.g. BoucWenModel) need RAW (un-normalized) u/y0 as input -- feeding
    them z-scored input while their physics constants stay at true SI
    scale causes a severe, silent unit mismatch (found during search:
    the model's honest physical prediction becomes ~4 orders of
    magnitude smaller than the normalized target, indistinguishable from
    an untrained all-zero output). Any model that sets
    `self.requires_raw_io = True` gets routed through the raw-I/O path
    below automatically; this is a general mechanism; not specific to
    BoucWenModel, so any FUTURE physically-parameterized model added to
    this project inherits the same correct handling by just setting the
    same flag, rather than needing this bug rediscovered per model."""
    return bool(getattr(model, "requires_raw_io", False))


def call_model(model, sequence_norm, sequence_raw, normalizer, device):
    """Calls the model with whichever of (raw, normalized) u/y0 it
    actually needs, and returns (y_hat_norm, y_hat_raw, hidden_state) --
    ALWAYS both, computed consistently, so callers never need to branch
    on model type themselves. `BoucWenModel` additionally needs
    `sampling_time` (to convert to the true RK4 dt); other model types
    ignore it."""
    kwargs = {}
    if hasattr(model, "num_substeps"):
        kwargs["sampling_time"] = sequence_raw.sampling_time

    if model_requires_raw_io(model):
        y_hat_raw, hidden_state = model(sequence_raw.u.to(device), sequence_raw.y0.to(device), **kwargs)
        y_hat_norm = normalizer.normalize_y_tensor(y_hat_raw)
    else:
        y_hat_norm, hidden_state = model(sequence_norm.u.to(device), sequence_norm.y0.to(device), **kwargs)
        y_hat_raw = normalizer.denormalize_y_tensor(y_hat_norm)
    return y_hat_norm, y_hat_raw, hidden_state


def predict_sequence(model, sequence_norm, sequence_raw, normalizer, device):
    model.eval()
    with torch.no_grad():
        y_hat_norm, y_hat_raw, _ = call_model(model, sequence_norm, sequence_raw, normalizer, device)
    return y_hat_norm.detach().cpu().numpy()[0], y_hat_raw.detach().cpu().numpy()[0]


def aggregate_metrics_across_sequences(model, raw_sequences, norm_sequences, normalizer, device):
    all_targets_norm, all_predictions_norm = [], []
    all_targets_raw, all_predictions_raw = [], []
    model.eval()
    with torch.no_grad():
        for raw_sequence, norm_sequence in zip(raw_sequences, norm_sequences):
            prediction_norm, prediction_raw = predict_sequence(model, norm_sequence, raw_sequence, normalizer, device)
            all_targets_norm.append(norm_sequence.y[0].detach().cpu().numpy()[raw_sequence.warmup:])
            all_predictions_norm.append(prediction_norm[raw_sequence.warmup:])
            all_targets_raw.append(raw_sequence.y[0].detach().cpu().numpy()[raw_sequence.warmup:])
            all_predictions_raw.append(prediction_raw[raw_sequence.warmup:])
    return {
        "rmse_norm": prepare.rmse(np.concatenate(all_targets_norm, axis=0), np.concatenate(all_predictions_norm, axis=0)),
        "rmse_raw": prepare.rmse(np.concatenate(all_targets_raw, axis=0), np.concatenate(all_predictions_raw, axis=0)),
    }


def compute_training_loss(model, batched_window_norm_raw, normalizer, device):
    """`batched_window_norm_raw`: a SINGLE (sequence_norm, sequence_raw)
    pair whose batch dimension already holds `crops_per_step` windows
    (see prepare.sample_batched_training_windows) -- ONE forward/backward
    pass covers the whole batch, instead of `crops_per_step` sequential
    passes. Loss is always computed in NORMALIZED space (for
    comparability across model types), but the model itself is fed
    whichever of the pair it actually needs."""
    sequence_norm, sequence_raw = batched_window_norm_raw
    y_hat_norm, _, _ = call_model(model, sequence_norm, sequence_raw, normalizer, device)
    diff = y_hat_norm - sequence_norm.y.to(device)
    total_count = diff.numel()
    if total_count == 0:
        raise RuntimeError("Training loss could not be computed because no training sequences were provided.")
    return torch.sum(diff ** 2) / total_count


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_fold(validation_fold: str, all_train_sequences: dict[str, "prepare.BoucWenSequence"], general_config: dict):
    torch.set_num_threads(general_config.get("threads_per_worker", torch.get_num_threads()))
    prepare.set_global_seed(general_config["seed"])
    window_rng = np.random.default_rng(general_config["seed"] + hash(validation_fold) % 100000)

    raw_validation = all_train_sequences[validation_fold]
    raw_train_list = [s for name, s in all_train_sequences.items() if name != validation_fold]

    history_window = int(general_config["history_window"])
    window_min_length = int(general_config["window_min_length"])
    normalizer = prepare.Normalizer.fit(raw_train_list, history_window=history_window)
    train_list_norm = [normalizer.normalize_sequence(s) for s in raw_train_list]
    validation_norm = normalizer.normalize_sequence(raw_validation)

    # One shared, continuous self-generated recording (see prepare.py) --
    # folds are contiguous index ranges within it, matching the CED/EMPS
    # pattern. Training windows are sampled from the ranges belonging to
    # the OTHER (non-held-out) folds only.
    u_full, y_full, sampling_time = prepare.load_raw_sequence_array("train_full")
    # Build valid_ranges directly from each non-held-out fold's own
    # (start, stop) span within the shared array.
    valid_ranges = []
    cursor = 0
    total_length = len(u_full)
    num_folds = len([s for s in all_train_sequences.values()])
    fold_length = total_length // num_folds
    for i, name in enumerate(sorted(all_train_sequences.keys())):
        start = i * fold_length
        stop = total_length if i == num_folds - 1 else (i + 1) * fold_length
        if name != validation_fold:
            valid_ranges.append((start, stop))

    extend_windows_to_range_end = bool(config_pars.get("extend_windows_to_range_end", False))
    crops_per_step = max(1, int(config_pars.get("crops_per_step", 1)))

    def sample_fresh_training_windows():
        window_raw = prepare.sample_batched_training_windows(
            u_full=u_full, y_full=y_full, sampling_time=sampling_time, history_window=history_window,
            valid_ranges=valid_ranges, min_window_length=window_min_length, rng=window_rng,
            batch_size=crops_per_step, extend_to_range_end=extend_windows_to_range_end,
        )
        return normalizer.normalize_sequence(window_raw), window_raw

    device = config_pars.get("device_override") or general_config["device"]
    checkpoint_root = Path(general_config["checkpoint_path"])
    log_dir = Path(general_config["log_dir"])
    fold_dir = checkpoint_root / validation_fold
    fold_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config_pars, general_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_pars["lr"], weight_decay=config_pars.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, threshold=1e-4, min_lr=1e-5)

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_evaluations = 0
    fold_start_time = time.perf_counter()

    _time_budget_override = config_pars.get("fold_time_budget_seconds")
    fold_time_budget_seconds = float(_time_budget_override if _time_budget_override is not None else general_config["search_time_budget_seconds"])
    _eval_every_override = config_pars.get("eval_every")
    eval_every = int(_eval_every_override if _eval_every_override is not None else general_config["eval_every"])

    fold_log_path = log_dir / f"train_{validation_fold}.log"
    initialize_fold_log(fold_log_path, validation_fold)

    initial_train_metrics = aggregate_metrics_across_sequences(model, raw_train_list, train_list_norm, normalizer, device)
    initial_validation_metrics = aggregate_metrics_across_sequences(model, [raw_validation], [validation_norm], normalizer, device)
    with torch.no_grad():
        initial_loss = compute_training_loss(model, sample_fresh_training_windows(), normalizer, device=device)
    best_val_rmse_norm = float(initial_validation_metrics["rmse_norm"])
    elapsed = time.perf_counter() - fold_start_time
    initial_line = f"0,{elapsed:.3f},{initial_loss.item():.6f},{initial_train_metrics['rmse_norm']:.6f},{initial_validation_metrics['rmse_norm']:.6f}"
    print(f"[{validation_fold}] {initial_line}  (epochs/sec: n/a)")
    append_log_line(fold_log_path, initial_line)
    prev_epoch, prev_elapsed = 0, elapsed

    skipped_nan_steps = 0
    for epoch in range(1, config_pars["max_epochs"] + 1):
        if time.perf_counter() - fold_start_time >= fold_time_budget_seconds:
            break
        model.train()
        optimizer.zero_grad()
        batched_window = sample_fresh_training_windows()
        loss = compute_training_loss(model, batched_window, normalizer, device=device)
        loss.backward()
        grad_clip = config_pars.get("grad_clip_norm", 0.0)
        if grad_clip and grad_clip > 0.0:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            total_grad_norm = torch.norm(torch.stack([
                torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None
            ])) if any(p.grad is not None for p in model.parameters()) else torch.tensor(0.0)

        if not torch.isfinite(total_grad_norm):
            skipped_nan_steps += 1
            optimizer.zero_grad()
            continue
        optimizer.step()

        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == config_pars["max_epochs"]
        if not should_evaluate:
            continue
        if time.perf_counter() - fold_start_time >= fold_time_budget_seconds:
            break

        train_metrics = aggregate_metrics_across_sequences(model, raw_train_list, train_list_norm, normalizer, device)
        validation_metrics = aggregate_metrics_across_sequences(model, [raw_validation], [validation_norm], normalizer, device)
        elapsed = time.perf_counter() - fold_start_time
        line = f"{epoch},{elapsed:.3f},{loss.item():.6f},{train_metrics['rmse_norm']:.6f},{validation_metrics['rmse_norm']:.6f}"
        d_epoch = epoch - prev_epoch
        d_time = elapsed - prev_elapsed
        epochs_per_sec = d_epoch / d_time if d_time > 0 else float("nan")
        print(f"[{validation_fold}] {line}  (epochs/sec: {epochs_per_sec:.3f})")
        append_log_line(fold_log_path, line)
        prev_epoch, prev_elapsed = epoch, elapsed

        if validation_metrics["rmse_norm"] < best_val_rmse_norm:
            best_val_rmse_norm = float(validation_metrics["rmse_norm"])
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            stale_evaluations = 0
        else:
            stale_evaluations += 1
        scheduler.step(validation_metrics["rmse_norm"])
        if stale_evaluations >= config_pars["early_stopping_patience"]:
            break

    model.load_state_dict(best_state_dict)
    final_train_metrics = aggregate_metrics_across_sequences(model, raw_train_list, train_list_norm, normalizer, device)
    final_validation_metrics = aggregate_metrics_across_sequences(model, [raw_validation], [validation_norm], normalizer, device)

    checkpoint_payload = {
        "validation_fold": validation_fold, "model_state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config_pars), "general_config": general_config, "normalizer": normalizer.state_dict(),
        "best_epoch": best_epoch,
        "metrics": {
            "train_rmse_norm": float(final_train_metrics["rmse_norm"]), "validation_rmse_norm": float(final_validation_metrics["rmse_norm"]),
            "train_rmse_raw": float(final_train_metrics["rmse_raw"]), "validation_rmse_raw": float(final_validation_metrics["rmse_raw"]),
        },
    }
    torch.save(checkpoint_payload, fold_dir / "model.pt")

    return {
        "validation_name": validation_fold, "best_epoch": best_epoch, "checkpoint_path": str(fold_dir / "model.pt"),
        "log_path": str(fold_log_path), "train_rmse_norm": float(final_train_metrics["rmse_norm"]),
        "validation_rmse_norm": float(final_validation_metrics["rmse_norm"]), "train_rmse_raw": float(final_train_metrics["rmse_raw"]),
        "validation_rmse_raw": float(final_validation_metrics["rmse_raw"]), "skipped_nan_steps": skipped_nan_steps,
    }


def summarize_folds(fold_results):
    train_values_norm = np.asarray([r["train_rmse_norm"] for r in fold_results], dtype=np.float64)
    validation_values_norm = np.asarray([r["validation_rmse_norm"] for r in fold_results], dtype=np.float64)
    train_values_raw = np.asarray([r["train_rmse_raw"] for r in fold_results], dtype=np.float64)
    validation_values_raw = np.asarray([r["validation_rmse_raw"] for r in fold_results], dtype=np.float64)
    best_epochs = np.asarray([r["best_epoch"] for r in fold_results], dtype=np.float64)
    return {
        "metric": "rmse", "num_folds": len(fold_results),
        "train_rmse_norm_mean": float(train_values_norm.mean()), "train_rmse_norm_std": float(train_values_norm.std()),
        "validation_rmse_norm_mean": float(validation_values_norm.mean()), "validation_rmse_norm_std": float(validation_values_norm.std()),
        "train_rmse_raw_mean": float(train_values_raw.mean()), "train_rmse_raw_std": float(train_values_raw.std()),
        "validation_rmse_raw_mean": float(validation_values_raw.mean()), "validation_rmse_raw_std": float(validation_values_raw.std()),
        "best_epoch_mean": float(best_epochs.mean()), "best_epoch_median": int(np.median(best_epochs)), "folds": fold_results,
    }


def save_current_summary(checkpoint_root: Path, summary: dict) -> Path:
    summary_path = checkpoint_root / "cross_validation_summary.json"
    write_json(summary_path, summary)
    return summary_path


def maybe_update_best_so_far(checkpoint_root: Path, summary: dict) -> bool:
    best_root = checkpoint_root / "best_so_far"
    best_summary_path = best_root / "cross_validation_summary.json"
    current_value = float(summary["validation_rmse_norm_mean"])
    if best_summary_path.exists():
        previous_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        if current_value >= float(previous_summary["validation_rmse_norm_mean"]):
            return False
    updated_folds = []
    for fold_result in summary["folds"]:
        validation_name = str(fold_result["validation_name"])
        target_checkpoint = best_root / validation_name / "model.pt"
        target_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(str(fold_result["checkpoint_path"])), target_checkpoint)
        updated_fold_result = dict(fold_result)
        updated_fold_result["checkpoint_path"] = str(target_checkpoint)
        updated_folds.append(updated_fold_result)
    best_summary = dict(summary)
    best_summary["folds"] = updated_folds
    write_json(best_summary_path, best_summary)
    return True


def plot_cross_validation_curves(fold_results, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(fold_results), 1)))
    for color, fold_result in zip(colors, fold_results):
        log_path = Path(str(fold_result["log_path"]))
        if not log_path.exists():
            continue
        epochs, window_losses, val_rmses = [], [], []
        with log_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        for row in rows[2:]:
            if len(row) < 5:
                continue
            epochs.append(int(row[0])); window_losses.append(float(row[2])); val_rmses.append(float(row[4]))
        label = str(fold_result["validation_name"])
        axes[0].plot(epochs, window_losses, linewidth=1.0, color=color, alpha=0.8, label=label)
        axes[1].plot(epochs, val_rmses, linewidth=1.4, color=color, label=label)
    axes[0].set_ylabel("random-window training loss (normalized MSE)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("held-out fold RMSE (normalized)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("Leave-one-fold-out training curves (each color = one held-out fold)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main():
    train_sequences, general_config = prepare.load_train_sequences_and_config()
    prepare.set_global_seed(general_config["seed"])

    checkpoint_root = Path(general_config["checkpoint_path"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(general_config["plots_path"])

    fold_names = sorted(train_sequences.keys())
    n_workers = len(fold_names)
    general_config["threads_per_worker"] = max(1, torch.get_num_threads() // n_workers)

    # NOTE on GPU: all 5 fold workers below share whatever single device
    # general_config["device"] resolves to (prepare.py auto-detects CUDA
    # if available). `spawn` (used here) is the CUDA-safe multiprocessing
    # start method (unlike `fork`, which is NOT safe with an already-
    # initialized CUDA context) -- multiple processes CAN share one GPU
    # correctly via CUDA's own context switching, but if 5 processes
    # simultaneously training on it causes memory pressure or contention-
    # driven slowdown on a given GPU, reduce concurrency (e.g. run folds
    # sequentially, or fewer at a time) rather than assuming more
    # parallelism is always faster on GPU the way it reliably is on CPU.
    mp_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(train_one_fold, name, train_sequences, general_config): name for name in fold_names}
        fold_results = [f.result() for f in as_completed(futures)]
    fold_results.sort(key=lambda r: r["validation_name"])

    summary = summarize_folds(fold_results)
    save_current_summary(checkpoint_root, summary)
    best_so_far_updated = maybe_update_best_so_far(checkpoint_root, summary)
    plot_cross_validation_curves(fold_results, plots_dir / "cross_validation_curves.png")

    print("")
    print("Cross-validation summary")
    print(f"Mean train RMSE norm      : {summary['train_rmse_norm_mean']:.6f}")
    print(f"Mean validation RMSE norm : {summary['validation_rmse_norm_mean']:.6f}")
    print(f"Median best epoch         : {summary['best_epoch_median']}")
    print(f"Best-so-far checkpoint    : {'updated' if best_so_far_updated else 'kept previous'}")
    print(f"Training-curve plot       : {plots_dir / 'cross_validation_curves.png'}")
    total_skipped = sum(int(r.get("skipped_nan_steps", 0)) for r in fold_results)
    if total_skipped > 0:
        print(f"NOTE: {total_skipped} step(s) across all folds had a non-finite gradient and were skipped.")
    return summary


if __name__ == "__main__":
    main()