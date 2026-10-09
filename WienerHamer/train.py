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
    "lr": 3e-3,
    "max_epochs": 3000,
    "type": "WHHYBRID",
    "n_taps_g1": 128,
    "n_taps_g2": 128,
    "residual_hidden_states": 24,
    "residual_recurrent": "GRU",
    "n_hidden_states": 64,
    "hidden_sizes": [16],
    "activation": "Tanh",
    "num_layers": 1,
    "dropout_prob": 0.0,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 100,
    "direct_feedthrough": False,
    "extend_windows_to_range_end": False,
    "crops_per_step": 1,
    # None -> use prepare.py's shared defaults (see the note in the BoucWen
    # project's train.py for why per-experiment overrides matter for slow
    # architectures).
    "fold_time_budget_seconds": None,
    "eval_every": None,
}

def append_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def initialize_fold_log(log_path: Path, fold_name: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{fold_name}\n")
        handle.write("epoch,window_loss_norm_mse,train_rmse_norm,val_rmse_norm\n")


def compute_rmse(y_true, y_pred, warmup=0) -> float:
    return prepare.rmse(y_true[warmup:], y_pred[warmup:])


def build_model(model_config, general_config):
    return build_model_from_config(
        config_pars=model_config, n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"], n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])


def predict_sequence(model, normalized_sequence, normalizer, device):
    model.eval()
    with torch.no_grad():
        y_hat_norm, _ = model(normalized_sequence.u.to(device), normalized_sequence.y0.to(device))
        y_hat_raw = normalizer.denormalize_y_tensor(y_hat_norm)
    return y_hat_norm.detach().cpu().numpy()[0], y_hat_raw.detach().cpu().numpy()[0]


def evaluate_one_sequence(model, sequence_raw, sequence_norm, normalizer, device):
    prediction_norm, prediction_raw = predict_sequence(model, sequence_norm, normalizer, device)
    target_norm = sequence_norm.y[0].detach().cpu().numpy()
    target_raw = sequence_raw.y[0].detach().cpu().numpy()
    return {
        "rmse_norm": compute_rmse(target_norm, prediction_norm, warmup=sequence_raw.warmup),
        "rmse_raw": compute_rmse(target_raw, prediction_raw, warmup=sequence_raw.warmup),
    }


def aggregate_metrics_across_sequences(model, raw_sequences, norm_sequences, normalizer, device):
    all_targets_norm, all_predictions_norm = [], []
    all_targets_raw, all_predictions_raw = [], []
    model.eval()
    with torch.no_grad():
        for raw_sequence, norm_sequence in zip(raw_sequences, norm_sequences):
            prediction_norm, prediction_raw = predict_sequence(model, norm_sequence, normalizer, device)
            all_targets_norm.append(norm_sequence.y[0].detach().cpu().numpy()[raw_sequence.warmup:])
            all_predictions_norm.append(prediction_norm[raw_sequence.warmup:])
            all_targets_raw.append(raw_sequence.y[0].detach().cpu().numpy()[raw_sequence.warmup:])
            all_predictions_raw.append(prediction_raw[raw_sequence.warmup:])
    return {
        "rmse_norm": prepare.rmse(np.concatenate(all_targets_norm, axis=0), np.concatenate(all_predictions_norm, axis=0)),
        "rmse_raw": prepare.rmse(np.concatenate(all_targets_raw, axis=0), np.concatenate(all_predictions_raw, axis=0)),
    }


def compute_training_loss(model, normalized_sequences, device):
    total_sse, total_count = None, 0
    for sequence in normalized_sequences:
        y_hat, _ = model(sequence.u.to(device), sequence.y0.to(device))
        diff = y_hat - sequence.y.to(device)
        sse = torch.sum(diff ** 2)
        total_sse = sse if total_sse is None else total_sse + sse
        total_count += diff.numel()
    if total_sse is None or total_count == 0:
        raise RuntimeError("Training loss could not be computed because no training sequences were provided.")
    return total_sse / total_count


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_fold(validation_name, train_sequences_raw, general_config):
    torch.set_num_threads(general_config.get("threads_per_worker", torch.get_num_threads()))
    prepare.set_global_seed(general_config["seed"])
    window_rng = np.random.default_rng(general_config["seed"] + hash(validation_name) % 100000)

    train_names = [name for name in train_sequences_raw if name != validation_name]
    raw_train_list = [train_sequences_raw[name] for name in train_names]
    raw_validation = train_sequences_raw[validation_name]

    history_window = int(general_config["history_window"])
    window_min_length = int(general_config["window_min_length"])
    normalizer = prepare.Normalizer.fit(raw_train_list, history_window=history_window)
    train_list_norm = [normalizer.normalize_sequence(s) for s in raw_train_list]
    validation_norm = normalizer.normalize_sequence(raw_validation)

    u_full, y_full, sampling_time = prepare.load_raw_train_arrays()
    valid_ranges = [(s.start_index, s.stop_index) for s in raw_train_list]
    extend_windows_to_range_end = bool(config_pars.get("extend_windows_to_range_end", False))
    crops_per_step = max(1, int(config_pars.get("crops_per_step", 1)))

    def sample_fresh_training_windows():
        windows = []
        for _ in range(crops_per_step):
            window = prepare.sample_training_window(
                u_full=u_full, y_full=y_full, sampling_time=sampling_time, history_window=history_window,
                valid_ranges=valid_ranges, min_window_length=window_min_length, rng=window_rng,
                extend_to_range_end=extend_windows_to_range_end,
            )
            windows.append(normalizer.normalize_sequence(window))
        return windows

    device = general_config["device"]
    checkpoint_root = Path(general_config["checkpoint_path"])
    log_dir = Path(general_config["log_dir"])
    fold_dir = checkpoint_root / validation_name
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

    fold_log_path = log_dir / f"train_{validation_name}.log"
    initialize_fold_log(fold_log_path, validation_name)

    initial_train_metrics = aggregate_metrics_across_sequences(model, raw_train_list, train_list_norm, normalizer, device)
    initial_validation_metrics = evaluate_one_sequence(model, raw_validation, validation_norm, normalizer, device)
    with torch.no_grad():
        initial_loss = compute_training_loss(model, sample_fresh_training_windows(), device=device)
    best_val_rmse_norm = float(initial_validation_metrics["rmse_norm"])
    initial_line = f"0,{initial_loss.item():.6f},{initial_train_metrics['rmse_norm']:.6f},{initial_validation_metrics['rmse_norm']:.6f}"
    print(f"[{validation_name}] {initial_line}")
    append_log_line(fold_log_path, initial_line)

    for epoch in range(1, config_pars["max_epochs"] + 1):
        if time.perf_counter() - fold_start_time >= fold_time_budget_seconds:
            break
        model.train()
        optimizer.zero_grad()
        window_list_norm = sample_fresh_training_windows()
        loss = compute_training_loss(model, window_list_norm, device=device)
        loss.backward()
        grad_clip = config_pars.get("grad_clip_norm", 0.0)
        if grad_clip and grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == config_pars["max_epochs"]
        if not should_evaluate:
            continue
        if time.perf_counter() - fold_start_time >= fold_time_budget_seconds:
            break

        train_metrics = aggregate_metrics_across_sequences(model, raw_train_list, train_list_norm, normalizer, device)
        validation_metrics = evaluate_one_sequence(model, raw_validation, validation_norm, normalizer, device)
        line = f"{epoch},{loss.item():.6f},{train_metrics['rmse_norm']:.6f},{validation_metrics['rmse_norm']:.6f}"
        print(f"[{validation_name}] {line}")
        append_log_line(fold_log_path, line)

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
    final_validation_metrics = evaluate_one_sequence(model, raw_validation, validation_norm, normalizer, device)

    checkpoint_payload = {
        "validation_name": validation_name, "train_names": train_names, "model_state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config_pars), "general_config": general_config, "normalizer": normalizer.state_dict(),
        "best_epoch": best_epoch,
        "metrics": {
            "train_rmse_norm": float(final_train_metrics["rmse_norm"]), "validation_rmse_norm": float(final_validation_metrics["rmse_norm"]),
            "train_rmse_raw": float(final_train_metrics["rmse_raw"]), "validation_rmse_raw": float(final_validation_metrics["rmse_raw"]),
        },
    }
    torch.save(checkpoint_payload, fold_dir / "model.pt")

    return {
        "validation_name": validation_name, "best_epoch": best_epoch, "checkpoint_path": str(fold_dir / "model.pt"),
        "log_path": str(fold_log_path), "train_rmse_norm": float(final_train_metrics["rmse_norm"]),
        "validation_rmse_norm": float(final_validation_metrics["rmse_norm"]), "train_rmse_raw": float(final_train_metrics["rmse_raw"]),
        "validation_rmse_raw": float(final_validation_metrics["rmse_raw"]),
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
            if len(row) < 4:
                continue
            epochs.append(int(row[0])); window_losses.append(float(row[1])); val_rmses.append(float(row[3]))
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

    fold_names = sorted(train_sequences)
    n_workers = len(fold_names)
    general_config["threads_per_worker"] = max(1, torch.get_num_threads() // n_workers)

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
    return summary


if __name__ == "__main__":
    main()