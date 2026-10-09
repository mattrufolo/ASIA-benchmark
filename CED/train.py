from __future__ import annotations

import copy
import csv
import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import prepare
from model import build_model_from_config


# This is the ONLY dictionary an experiment should touch during the SEARCH
# phase (step 1). It is trained on prepare.load_search_train_sequences() and
# validated on prepare.load_search_val_sequences() -- see program.md for the
# full search/refit/test procedure and the rules governing this file.
config_pars = {
    "lr": 1e-3,
    "max_epochs": 3000,
    "type": "PHYSICS_TCN",
    "n_ode_states": 3,
    "resonant_init": True,
    "alpha_init": 2.0,
    "xi_init": 0.1,
    "omega0_init": 25.0,
    "sampling_time": 0.02,
    "tcn_channels": 16,
    "kernel_size": 3,
    "num_blocks": 3,
    "context_hidden_sizes": [32],
    "hidden_sizes": [32, 32],
    "activation": "Tanh",
    "num_layers": 1,
    "dropout_prob": 0.0,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "early_stopping_patience": 10,
    "direct_feedthrough": True,
}


def append_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def initialize_log(log_path: Path, title: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("epoch,train_loss_norm_mse,train_rmse_norm,val_rmse_norm\n")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray, warmup: int = 0) -> float:
    return prepare.rmse(y_true[warmup:], y_pred[warmup:])


def build_model(model_config: dict, general_config: dict) -> torch.nn.Module:
    return build_model_from_config(
        config_pars=model_config,
        n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"],
        n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])


def predict_sequence(
    model: torch.nn.Module,
    normalized_sequence: prepare.CEDSequence,
    normalizer: prepare.Normalizer,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        y_hat_norm, _ = model(
            normalized_sequence.u.to(device),
            normalized_sequence.y0.to(device),
        )
        y_hat_raw = normalizer.denormalize_y_tensor(y_hat_norm)
    return y_hat_norm.detach().cpu().numpy()[0], y_hat_raw.detach().cpu().numpy()[0]


def aggregate_metrics_across_sequences(
    model: torch.nn.Module,
    raw_sequences: list[prepare.CEDSequence],
    norm_sequences: list[prepare.CEDSequence],
    normalizer: prepare.Normalizer,
    device: str,
) -> dict[str, float]:
    """RMSE aggregated (concatenated) across all given sequences -- used both
    for the two search-train sequences (low + high) and the two search-val
    sequences (low + high), so the reported metric is a single scalar over
    both regimes jointly, not an average over interchangeable folds."""
    all_targets_norm = []
    all_predictions_norm = []
    all_targets_raw = []
    all_predictions_raw = []

    model.eval()
    with torch.no_grad():
        for raw_sequence, norm_sequence in zip(raw_sequences, norm_sequences):
            prediction_norm, prediction_raw = predict_sequence(
                model=model,
                normalized_sequence=norm_sequence,
                normalizer=normalizer,
                device=device,
            )
            all_targets_norm.append(norm_sequence.y[0].detach().cpu().numpy()[raw_sequence.warmup :])
            all_predictions_norm.append(prediction_norm[raw_sequence.warmup :])
            all_targets_raw.append(raw_sequence.y[0].detach().cpu().numpy()[raw_sequence.warmup :])
            all_predictions_raw.append(prediction_raw[raw_sequence.warmup :])

    return {
        "rmse_norm": prepare.rmse(
            np.concatenate(all_targets_norm, axis=0),
            np.concatenate(all_predictions_norm, axis=0),
        ),
        "rmse_raw": prepare.rmse(
            np.concatenate(all_targets_raw, axis=0),
            np.concatenate(all_predictions_raw, axis=0),
        ),
    }


def compute_training_loss(
    model: torch.nn.Module,
    normalized_sequences: list[prepare.CEDSequence],
    device: str,
) -> torch.Tensor:
    total_sse = None
    total_count = 0

    for sequence in normalized_sequences:
        y_hat, _ = model(sequence.u.to(device), sequence.y0.to(device))
        diff = y_hat - sequence.y.to(device)
        sse = torch.sum(diff**2)
        total_sse = sse if total_sse is None else total_sse + sse
        total_count += diff.numel()

    if total_sse is None or total_count == 0:
        raise RuntimeError("Training loss could not be computed because no training sequences were provided.")

    return total_sse / total_count


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_search_training_curve(log_path: Path, best_epoch: int, plot_path: Path) -> None:
    """Reads logs/train_search.log (written every `eval_every` epochs) and
    plots the training-window loss, the full-window train RMSE, and the
    search-val RMSE vs. epoch, with the chosen best_epoch marked. Since a
    fresh random window is sampled every step, `window_loss_norm_mse` is
    naturally noisier epoch-to-epoch than a fixed-batch loss would be --
    that noise is expected, not a sign of instability; `train_rmse_norm`
    (evaluated on the full, fixed [history_window:search_train_stop]
    window) is the smoother curve to read the overall training trend from.
    """
    epochs, window_losses, train_rmses, val_rmses = [], [], [], []
    with log_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    for row in rows[2:]:  # skip the title line and the CSV header line
        if len(row) < 4:
            continue
        epochs.append(int(row[0]))
        window_losses.append(float(row[1]))
        train_rmses.append(float(row[2]))
        val_rmses.append(float(row[3]))

    if not epochs:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(epochs, window_losses, linewidth=1.0, color="#adb5bd", label="random-window training loss (noisy, per-step)")
    axes[0].set_ylabel("training loss (normalized MSE)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(epochs, train_rmses, linewidth=1.6, color="#2a9d8f", label="train RMSE (full fixed window)")
    axes[1].plot(epochs, val_rmses, linewidth=1.8, color="#e76f51", label="search-val RMSE ([search_train_stop:400])")
    if best_epoch in epochs:
        axes[1].axvline(best_epoch, color="black", linestyle="--", linewidth=1.0, label=f"chosen best_epoch={best_epoch}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE (normalized)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Search-phase training curves")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def maybe_update_best_so_far(checkpoint_root: Path, summary: dict[str, object]) -> bool:
    """Bookkeeping across SEARCH iterations only (not related to the REFIT
    step). Keeps a copy of the single best search checkpoint seen so far,
    purely as a convenience/rollback aid during architecture search."""
    best_root = checkpoint_root / "best_so_far"
    best_summary_path = best_root / "search_summary.json"
    current_value = float(summary["validation_rmse_norm"])

    if best_summary_path.exists():
        previous_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        previous_value = float(previous_summary["validation_rmse_norm"])
        if current_value >= previous_value:
            return False

    best_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_root / "search" / "model.pt", best_root / "model.pt")
    write_json(best_summary_path, summary)
    return True


def main() -> dict[str, object]:
    search_train_sequences, search_val_sequences, general_config = prepare.load_search_datasets_and_config()
    prepare.set_global_seed(general_config["seed"])
    window_rng = np.random.default_rng(general_config["seed"])

    raw_train_list = list(search_train_sequences.values())
    raw_val_list = list(search_val_sequences.values())

    history_window = int(general_config["history_window"])
    search_train_stop = int(general_config["search_train_stop"])
    window_min_length = int(general_config["window_min_length"])

    # Normalizer statistics are fit ONCE on the full, fixed search-train data
    # (not on individual random windows), so normalization stays stable across
    # training steps even though the window sampled at each step changes.
    normalizer = prepare.Normalizer.fit(raw_train_list, history_window=history_window)
    full_train_list_norm = [normalizer.normalize_sequence(sequence) for sequence in raw_train_list]
    val_list_norm = [normalizer.normalize_sequence(sequence) for sequence in raw_val_list]

    # Raw arrays per realization, used to sample a NEW random training window
    # (random start index AND length) every training step. This is the fix
    # for the initial-condition network only ever seeing one fixed context
    # per regime -- see CED_description.md Section 6 / prepare.py's
    # `sample_training_window` docstring for the full rationale.
    raw_arrays_by_realization = {
        realization: prepare.load_raw_train_arrays(realization) for realization in prepare.REALIZATIONS
    }

    def sample_fresh_training_windows() -> list[prepare.CEDSequence]:
        windows = []
        for realization in prepare.REALIZATIONS:
            u_full, y_full, sampling_time = raw_arrays_by_realization[realization]
            window = prepare.sample_training_window(
                u_full=u_full,
                y_full=y_full,
                sampling_time=sampling_time,
                realization=realization,
                history_window=history_window,
                stop_index=search_train_stop,
                min_window_length=window_min_length,
                rng=window_rng,
            )
            windows.append(normalizer.normalize_sequence(window))
        return windows

    device = general_config["device"]
    checkpoint_root = Path(general_config["checkpoint_path"])
    search_dir = checkpoint_root / "search"
    search_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(general_config["log_dir"])
    plots_dir = Path(general_config["plots_path"])

    model = build_model(config_pars, general_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_pars["lr"],
        weight_decay=config_pars.get("weight_decay", 0.0),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        threshold=1e-4,
        min_lr=1e-5,
    )

    # Keep a valid fallback from the start so a time-budget stop still returns
    # the best checkpoint reached so far, even if training ends early.
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_evaluations = 0
    search_start_time = time.perf_counter()
    time_budget_seconds = float(general_config["search_time_budget_seconds"])
    eval_every = int(general_config["eval_every"])

    log_path = log_dir / "train_search.log"
    initialize_log(log_path, "search (random-window train on [history_window:search_train_stop], val on [search_train_stop:400])")

    initial_train_metrics = aggregate_metrics_across_sequences(
        model=model, raw_sequences=raw_train_list, norm_sequences=full_train_list_norm, normalizer=normalizer, device=device,
    )
    initial_val_metrics = aggregate_metrics_across_sequences(
        model=model, raw_sequences=raw_val_list, norm_sequences=val_list_norm, normalizer=normalizer, device=device,
    )
    with torch.no_grad():
        initial_loss = compute_training_loss(model, sample_fresh_training_windows(), device=device)
    best_val_rmse_norm = float(initial_val_metrics["rmse_norm"])
    initial_line = (
        f"0,{initial_loss.item():.6f},{initial_train_metrics['rmse_norm']:.6f},{initial_val_metrics['rmse_norm']:.6f}"
    )
    print(f"[search] {initial_line}")
    append_log_line(log_path, initial_line)

    for epoch in range(1, config_pars["max_epochs"] + 1):
        if time.perf_counter() - search_start_time >= time_budget_seconds:
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

        if time.perf_counter() - search_start_time >= time_budget_seconds:
            break

        train_metrics = aggregate_metrics_across_sequences(
            model=model, raw_sequences=raw_train_list, norm_sequences=full_train_list_norm, normalizer=normalizer, device=device,
        )
        val_metrics = aggregate_metrics_across_sequences(
            model=model, raw_sequences=raw_val_list, norm_sequences=val_list_norm, normalizer=normalizer, device=device,
        )

        line = f"{epoch},{loss.item():.6f},{train_metrics['rmse_norm']:.6f},{val_metrics['rmse_norm']:.6f}"
        print(f"[search] {line}")
        append_log_line(log_path, line)

        if val_metrics["rmse_norm"] < best_val_rmse_norm:
            best_val_rmse_norm = float(val_metrics["rmse_norm"])
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            stale_evaluations = 0
        else:
            stale_evaluations += 1

        scheduler.step(val_metrics["rmse_norm"])

        if stale_evaluations >= config_pars["early_stopping_patience"]:
            break

    model.load_state_dict(best_state_dict)

    final_train_metrics = aggregate_metrics_across_sequences(
        model=model, raw_sequences=raw_train_list, norm_sequences=full_train_list_norm, normalizer=normalizer, device=device,
    )
    final_val_metrics = aggregate_metrics_across_sequences(
        model=model, raw_sequences=raw_val_list, norm_sequences=val_list_norm, normalizer=normalizer, device=device,
    )

    checkpoint_payload = {
        "stage": "search",
        "model_state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config_pars),
        "general_config": general_config,
        "normalizer": normalizer.state_dict(),
        "best_epoch": best_epoch,
        "metrics": {
            "train_rmse_norm": float(final_train_metrics["rmse_norm"]),
            "validation_rmse_norm": float(final_val_metrics["rmse_norm"]),
            "train_rmse_raw": float(final_train_metrics["rmse_raw"]),
            "validation_rmse_raw": float(final_val_metrics["rmse_raw"]),
        },
    }
    torch.save(checkpoint_payload, search_dir / "model.pt")

    summary = {
        "stage": "search",
        "metric": "rmse",
        "best_epoch": best_epoch,
        "train_rmse_norm": float(final_train_metrics["rmse_norm"]),
        "validation_rmse_norm": float(final_val_metrics["rmse_norm"]),
        "train_rmse_raw": float(final_train_metrics["rmse_raw"]),
        "validation_rmse_raw": float(final_val_metrics["rmse_raw"]),
        "checkpoint_path": str(search_dir / "model.pt"),
    }
    write_json(checkpoint_root / "search_summary.json", summary)
    best_so_far_updated = maybe_update_best_so_far(checkpoint_root, summary)
    plot_search_training_curve(log_path, best_epoch, plots_dir / "search_training_curve.png")

    print("")
    print("Search summary (decisive metric: validation_rmse_norm)")
    print(f"Train RMSE (norm)      : {summary['train_rmse_norm']:.6f}")
    print(f"Validation RMSE (norm) : {summary['validation_rmse_norm']:.6f}")
    print(f"Best epoch             : {summary['best_epoch']}")
    print(f"Best-so-far checkpoint : {'updated' if best_so_far_updated else 'kept previous'}")
    print(f"Training-curve plot    : {plots_dir / 'search_training_curve.png'}")
    print("Note: this checkpoint was trained on [history_window:search_train_stop] ONLY.")
    print("Run refit.py once a winning config is chosen, before running test.py.")

    return summary


if __name__ == "__main__":
    main()