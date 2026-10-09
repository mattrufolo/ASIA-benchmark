from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import prepare
from model import build_model_from_config
from train import build_model, compute_training_loss, aggregate_metrics_across_sequences, config_pars


def load_search_summary(checkpoint_root: Path, use_best_so_far: bool) -> dict:
    summary_path = (
        checkpoint_root / "best_so_far" / "search_summary.json"
        if use_best_so_far
        else checkpoint_root / "search_summary.json"
    )
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run `train.py` first to produce a search checkpoint "
            "before refitting."
        )
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 2 of the CED procedure: retrain the SAME config currently in "
            "train.py's `config_pars` on the full [history_window:400] training "
            "data (both regimes, no held-out split). Run this ONCE, after a "
            "winning config has been chosen using train.py's search-validation "
            "RMSE -- never to compare candidates.\n\n"
            "By default, training stops when the full-window training RMSE "
            "plateaus (patience-based, like train.py's val-RMSE early stopping, "
            "but watching TRAIN RMSE since no held-out data exists at this "
            "stage), bounded by --time-budget as a safety net. This is "
            "preferred over blindly copying search's best_epoch count, since "
            "both phases now use randomly-sampled training windows: search's "
            "best_epoch is itself noisy (depends on which random windows it "
            "happened to draw) and refit samples from a wider range ([10:400] "
            "vs search's [10:350]), so a literal epoch-count transplant is not "
            "a reliable proxy for 'trained until converged' here."
        )
    )
    parser.add_argument(
        "--use-best-so-far",
        action="store_true",
        help=(
            "Read search's summary from checkpoints/best_so_far/search_summary.json "
            "instead of checkpoints/search_summary.json, purely for the provenance info "
            "recorded alongside the refit result (which search run motivated this refit)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Force an explicit, fixed number of refit epochs, bypassing the plateau-based "
            "stopping criterion entirely (e.g. for quick manual experiments or exact "
            "reproducibility of a specific run)."
        ),
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=100,
        help=(
            "Minimum number of epochs to run before the plateau-based criterion is allowed "
            "to trigger an early stop (ignored if --epochs is given). Guards against stopping "
            "too early on noisy initial fluctuations."
        ),
    )
    return parser.parse_args()


def initialize_log(log_path: Path, title: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write("epoch,window_loss_norm_mse,train_rmse_norm\n")


def append_log_line(log_path: Path, line: str) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def plot_refit_training_curve(log_path: Path, best_epoch: int, plot_path: Path) -> None:
    epochs, window_losses, train_rmses = [], [], []
    with log_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    for row in rows[2:]:
        if len(row) < 3:
            continue
        epochs.append(int(row[0]))
        window_losses.append(float(row[1]))
        train_rmses.append(float(row[2]))

    if not epochs:
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(epochs, window_losses, linewidth=1.0, color="#adb5bd", label="random-window training loss (noisy, per-step)")
    axes[0].set_ylabel("training loss (normalized MSE)")
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(epochs, train_rmses, linewidth=1.8, color="#2a9d8f", label="full-window train RMSE (the plateau signal)")
    if best_epoch in epochs:
        axes[1].axvline(best_epoch, color="black", linestyle="--", linewidth=1.0, label=f"chosen best_epoch={best_epoch}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE (normalized)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle("Refit-phase training curves (no held-out data -- plateau on TRAIN RMSE)")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> dict[str, object]:
    args = parse_args()

    refit_train_sequences, _, general_config = prepare.load_refit_datasets_and_config()
    prepare.set_global_seed(general_config["seed"])
    window_rng = np.random.default_rng(general_config["seed"])

    checkpoint_root = Path(general_config["checkpoint_path"])
    plots_dir = Path(general_config["plots_path"])
    log_dir = Path(general_config["log_dir"])
    search_summary = load_search_summary(checkpoint_root, args.use_best_so_far)

    raw_train_list = list(refit_train_sequences.values())
    history_window = int(general_config["history_window"])
    window_min_length = int(general_config["window_min_length"])
    eval_every = int(general_config["eval_every"])

    # A fresh normalizer fit on ALL available training data (both regimes,
    # [history_window:400]) -- this is the FINAL model, so it should use every
    # sample available to it, not just the normalizer fit on search-train.
    normalizer = prepare.Normalizer.fit(raw_train_list, history_window=history_window)
    full_train_list_norm = [normalizer.normalize_sequence(sequence) for sequence in raw_train_list]

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
                stop_index=len(u_full),  # full [history_window:400], no held-out split
                min_window_length=window_min_length,
                rng=window_rng,
            )
            windows.append(normalizer.normalize_sequence(window))
        return windows

    device = general_config["device"]
    model = build_model(config_pars, general_config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config_pars["lr"],
        weight_decay=config_pars.get("weight_decay", 0.0),
    )

    log_path = log_dir / "refit.log"
    initialize_log(log_path, "refit (random-window train on full [history_window:400], no held-out data)")

    fixed_epochs = args.epochs is not None
    max_epochs = int(args.epochs) if fixed_epochs else 20000  # generous safety cap for the plateau loop
    time_budget_seconds = float(general_config["refit_time_budget_seconds"])
    patience = int(general_config["refit_patience"])

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_train_rmse_norm = float("inf")
    stale_evaluations = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        initial_metrics = aggregate_metrics_across_sequences(
            model=model, raw_sequences=raw_train_list, norm_sequences=full_train_list_norm, normalizer=normalizer, device=device,
        )
    best_train_rmse_norm = float(initial_metrics["rmse_norm"])
    initial_line = f"0,nan,{initial_metrics['rmse_norm']:.6f}"
    print(f"[refit] {initial_line}")
    append_log_line(log_path, initial_line)

    for epoch in range(1, max_epochs + 1):
        if not fixed_epochs and time.perf_counter() - start_time >= time_budget_seconds:
            print(f"[refit] time budget ({time_budget_seconds:.0f}s) reached at epoch {epoch}.")
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

        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == max_epochs
        if not should_evaluate:
            continue

        train_metrics = aggregate_metrics_across_sequences(
            model=model, raw_sequences=raw_train_list, norm_sequences=full_train_list_norm, normalizer=normalizer, device=device,
        )
        line = f"{epoch},{loss.item():.6f},{train_metrics['rmse_norm']:.6f}"
        print(f"[refit] {line}")
        append_log_line(log_path, line)

        if train_metrics["rmse_norm"] < best_train_rmse_norm:
            best_train_rmse_norm = float(train_metrics["rmse_norm"])
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            stale_evaluations = 0
        else:
            stale_evaluations += 1

        if not fixed_epochs and epoch >= args.min_epochs and stale_evaluations >= patience:
            print(f"[refit] full-window train RMSE plateaued (patience={patience}) at epoch {epoch}.")
            break

    model.load_state_dict(best_state_dict)
    final_train_metrics = aggregate_metrics_across_sequences(
        model=model, raw_sequences=raw_train_list, norm_sequences=full_train_list_norm, normalizer=normalizer, device=device,
    )

    final_dir = checkpoint_root / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plots_dir / "refit_training_curve.png"
    plot_refit_training_curve(log_path, best_epoch, plot_path)

    checkpoint_payload = {
        "stage": "final_refit",
        "model_state_dict": model.state_dict(),
        "model_config": copy.deepcopy(config_pars),
        "general_config": general_config,
        "normalizer": normalizer.state_dict(),
        "epochs_trained": epoch,
        "best_epoch": best_epoch,
        "stopping_mode": "fixed_epochs" if fixed_epochs else "train_rmse_plateau",
        "source_search_summary": search_summary,
        "metrics": {
            # This is TRAINING-SET fit (all data used for refit training), not a
            # validation metric -- there is no held-out data left at this stage.
            # It is reported only as a sanity check, not a decision signal.
            "refit_train_rmse_norm": float(final_train_metrics["rmse_norm"]),
            "refit_train_rmse_raw": float(final_train_metrics["rmse_raw"]),
        },
    }
    torch.save(checkpoint_payload, final_dir / "model.pt")

    summary = {
        "stage": "final_refit",
        "stopping_mode": "fixed_epochs" if fixed_epochs else "train_rmse_plateau",
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "search_best_epoch": int(search_summary["best_epoch"]),
        "search_validation_rmse_norm": float(search_summary["validation_rmse_norm"]),
        "refit_train_rmse_norm": float(final_train_metrics["rmse_norm"]),
        "refit_train_rmse_raw": float(final_train_metrics["rmse_raw"]),
        "checkpoint_path": str(final_dir / "model.pt"),
        "training_curve_plot": str(plot_path),
    }
    (final_dir / "refit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print("Refit summary")
    print(f"Stopping mode                                                          : {summary['stopping_mode']}")
    print(f"Epochs run (best_epoch={best_epoch})                                   : {epoch}")
    print(f"Refit train RMSE (norm, NOT a validation metric)                       : {summary['refit_train_rmse_norm']:.6f}")
    print(f"Saved final checkpoint                                                 : {final_dir / 'model.pt'}")
    print(f"Training-curve plot                                                    : {plot_path}")
    print("Run test.py now to get the official (reporting-only) test RMSE.")

    return summary


if __name__ == "__main__":
    main()