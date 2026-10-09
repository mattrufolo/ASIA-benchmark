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


# -----------------------------------------------------------------------------------
# config_pars: model architecture + training hyperparameters. CAN be modified freely
# during the ASIA search, together with model.py.
#
# Fixed-iteration training, no early stopping: training always runs exactly
# config_pars["max_epochs"] steps (or until `time_budget_seconds` -- a wall-clock
# SAFETY NET, not a metric-based stop -- is hit). There is no LR scheduler and no
# patience-based early stop. This is an explicit project invariant (do not
# reintroduce a validation-triggered stop): the goal is to let a small, fixed LR run
# for a long, fixed, predetermined number of steps so slow/small patterns in the
# EMPS trajectories have a real chance to be learned, and so that runs are directly
# comparable to each other (same stopping rule every time) rather than each having
# stopped at a different, metric-dependent point.
# -----------------------------------------------------------------------------------
config_pars = {
    "lr": 1e-3,
    "max_epochs": 20000,
    # Deliberately starts as a plain black-box baseline (matching Two_Tanks' own
    # literal starting config), NOT a model that won a previous search cycle --
    # avoid carrying forward an unearned prior about which family wins. Start cold:
    # black-box first, then white-box, then a combination -- see PROGRAM.md's Search
    # Strategy for the staged order across the required 15-20 iterations.
    "type": "LTC",
    "n_hidden_states": 32,
    "hidden_sizes": [32, 32],
    "activation": "Tanh",
    "num_layers":  1,
    "dropout_prob": 0.0,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    # OFF by default. For a system with a pure integrator, an UNTRAINED direct
    # `u -> y` shortcut starts out strongly correlated with the (noisy, bang-bang)
    # force input and can dominate the recurrent/physical part's own output early in
    # training -- measured directly: with this on, model output can correlate >0.95
    # with the raw input at initialization, producing predictions that visibly track
    # the input's shape instead of the smooth output. Turn back on deliberately (and
    # check plots/train_fit.png) if you want to test whether it helps once the rest
    # of the model has learned something real.
    "direct_feedthrough": False,
    "extend_windows_to_range_end": False,
    "crops_per_step": 1,
    # Fraction of each step's `crops_per_step` window draws that are forced to the
    # FULL first-80% span instead of a short window. 0.0 is fine for black-box types
    # (RNN/GRU/LSTM/LATENT): their hidden state is squashed through tanh/sigmoid
    # gates every step, which gives them natural, partial resistance to unbounded
    # drift. White-box/physical types (WIENER/TWOMASS/LUGRE) and HYBRID are
    # different: every training window's initial condition is built from the REAL
    # ground-truth u/y history right before it, so a small constant parameter bias
    # (e.g. a slightly wrong offset) barely moves a short window's loss -- it only
    # compounds into a large error once the model runs CONTINUOUSLY with no resets,
    # exactly what validation/test do. Measured directly: a 1N offset error (out of
    # a true offset of -3N) produced ~5500x more error under continuous simulation
    # than averaged over short, truth-anchored windows. Set this > 0 (try
    # 0.1-0.25) for those families so the optimizer is occasionally shown a
    # long/full window and can actually see (and fix) that bias. Most steps stay
    # cheap; the occasional long draw costs roughly as much as one full-sequence
    # epoch.
    "long_window_prob": 0.0,
    # Adds the least-squares SLOPE of the (y_hat - y_true) residual, computed over
    # EACH sampled window, to the loss. This only has teeth if the window is long
    # enough for a real bias to produce a visible slope -- pair with
    # `long_window_prob > 0` for white-box/HYBRID types; on short windows alone it
    # will barely register the exact failure mode described above.
    "drift_penalty_weight": 0.0,
    # No override: fall back to prepare.py's config_pars_general["time_budget_seconds"],
    # the single authoritative per-run wall-clock safety net for this search cycle.
    "time_budget_seconds": None,
}

# Model families that integrate with a Python per-timestep loop (see model.py)
# instead of a native/vectorized recurrent kernel -- meaningfully slower per epoch
# than RNN/GRU/LSTM at real EMPS sequence lengths. Used only to decide whether to
# print the time-budget warning below with an extra note.
SLOW_LOOP_MODEL_TYPES = {"WIENER", "LUGRE", "TWOMASS", "LATENT", "HYBRID"}


def append_log_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def initialize_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("epoch,window_loss_norm_mse,val_rmse_norm\n")


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray, warmup: int = 0) -> float:
    return prepare.rmse(y_true[warmup:], y_pred[warmup:])


def build_model(model_config: dict, general_config: dict) -> torch.nn.Module:
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


def evaluate_sequence(model, sequence_raw, sequence_norm, normalizer, device) -> dict:
    """Runs the model CONTINUOUSLY over the full sequence, but RMSE is
    computed only after `sequence_raw.warmup` samples -- for the
    validation sequence, this means the model simulates through the whole
    first 80% before ever being scored on the last 20%."""
    prediction_norm, prediction_raw = predict_sequence(model, sequence_norm, normalizer, device)
    target_norm = sequence_norm.y[0].detach().cpu().numpy()
    target_raw = sequence_raw.y[0].detach().cpu().numpy()
    return {
        "rmse_norm": compute_rmse(target_norm, prediction_norm, warmup=sequence_raw.warmup),
        "rmse_raw": compute_rmse(target_raw, prediction_raw, warmup=sequence_raw.warmup),
    }


def compute_training_loss(model, normalized_sequences, device, drift_penalty_weight=0.0) -> torch.Tensor:
    """Ordinary MSE (over the sampled training windows) plus an optional
    drift-penalty term. See config_pars's own comment: this term only helps if at
    least some of `normalized_sequences` are long enough (see `long_window_prob`)
    for a real bias to produce a visible residual slope."""
    total_sse = None
    total_count = 0
    total_drift_penalty = None
    num_sequences = 0

    for sequence in normalized_sequences:
        y_hat, _ = model(sequence.u.to(device), sequence.y0.to(device))
        diff = y_hat - sequence.y.to(device)
        sse = torch.sum(diff**2)
        total_sse = sse if total_sse is None else total_sse + sse
        total_count += diff.numel()
        num_sequences += 1

        if drift_penalty_weight > 0.0:
            T = diff.shape[1]
            if T > 1:
                t_idx = torch.arange(T, device=device, dtype=diff.dtype) - (T - 1) / 2.0
                t_idx = t_idx / max(float(t_idx.abs().max().item()), 1.0)
                residual = diff.squeeze(-1)
                numerator = (residual * t_idx.unsqueeze(0)).sum(dim=1)
                denominator = (t_idx**2).sum().clamp(min=1e-8)
                slope = numerator / denominator
                penalty = torch.sum(slope**2)
                total_drift_penalty = penalty if total_drift_penalty is None else total_drift_penalty + penalty

    if total_sse is None or total_count == 0:
        raise RuntimeError("Training loss could not be computed because no training sequences were provided.")

    loss = total_sse / total_count
    if drift_penalty_weight > 0.0 and total_drift_penalty is not None and num_sequences > 0:
        loss = loss + drift_penalty_weight * (total_drift_penalty / num_sequences)
    return loss


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def warn_if_time_budget_too_tight(seconds_per_epoch: float, max_epochs: int, time_budget_seconds: float, model_type: str) -> None:
    projected_total = seconds_per_epoch * max_epochs
    if projected_total <= time_budget_seconds:
        return
    epochs_reachable = int(time_budget_seconds / max(seconds_per_epoch, 1e-9))
    slow_note = (
        f" `{model_type}` integrates with a Python per-timestep loop, which is "
        "generally slower per epoch than the RNN/GRU/LSTM kernels." if model_type in SLOW_LOOP_MODEL_TYPES else ""
    )
    print(
        f"[train] WARNING: at ~{seconds_per_epoch:.3f}s/epoch, reaching max_epochs={max_epochs} would take "
        f"~{projected_total / 60.0:.1f} min, but time_budget_seconds={time_budget_seconds:.0f}s "
        f"(~{time_budget_seconds / 60.0:.1f} min) will cut the run short at ~epoch {epochs_reachable} instead."
        f"{slow_note} This is the wall-clock safety net, not early stopping, but it means this run will NOT "
        "reach the requested fixed number of iterations. Consider raising `time_budget_seconds` in prepare.py's "
        "config_pars_general (then re-run prepare.py) and/or lowering `max_epochs` for this model type."
    )


def plot_training_curve(log_path: Path, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        return
    epochs, window_losses, val_rmses = [], [], []
    with log_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    for row in rows[1:]:
        if len(row) < 3:
            continue
        epochs.append(int(row[0])); window_losses.append(float(row[1])); val_rmses.append(float(row[2]))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(epochs, window_losses, linewidth=1.0, color="#1d3557")
    axes[0].set_ylabel("training loss (normalized MSE + drift penalty)")
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss vs. epoch")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(epochs, val_rmses, linewidth=1.4, color="#e76f51")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation RMSE (normalized, last 20% only)")
    axes[1].set_title("Continuous-horizon validation RMSE vs. epoch")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def plot_fit_diagnostic(model, validation_raw, validation_norm, normalizer, device, plot_path: Path) -> None:
    """Model vs. true position over the WHOLE training trajectory (validation
    spans train+validation -- see prepare.py), with a vertical line marking where
    the held-out last 20% starts. Left of the line is data the optimizer directly
    trained on (via short windows drawn from it); if the fit is bad there too, the
    model did not learn from the training data at all, as opposed to a
    generalization problem where only the right side is bad. Regenerated fresh on
    every run so it can never go stale."""
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    _, prediction_raw = predict_sequence(model, validation_norm, normalizer, device)
    time_axis = np.arange(validation_raw.num_samples, dtype=np.float32) * validation_raw.sampling_time
    u_values = validation_raw.u[0].detach().cpu().numpy()
    y_true = validation_raw.y[0].detach().cpu().numpy()
    split_time = validation_raw.warmup * validation_raw.sampling_time

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_axis, u_values[:, 0], linewidth=0.8, color="#006d77")
    axes[0].set_ylabel(prepare.input_names[0])
    axes[0].set_title("Full training trajectory: input (motor force, N)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, y_true[:, 0], linewidth=2.0, color="#1d3557", label="true position")
    axes[1].plot(time_axis, prediction_raw[:, 0], linewidth=1.2, color="#e76f51", label="model prediction")
    axes[1].axvline(split_time, color="black", linestyle="--", linewidth=1.2, label="train / validation split")
    axes[1].set_ylabel(prepare.output_names[0] + " [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title("Fit: left of dashed line = trained on, right of dashed line = held out")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def maybe_update_best_so_far(checkpoint_root: Path, summary: dict) -> bool:
    best_root = checkpoint_root / "best_so_far"
    best_summary_path = best_root / "summary.json"
    current_value = float(summary["validation_rmse_norm"])
    if best_summary_path.exists():
        previous_summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        if current_value >= float(previous_summary["validation_rmse_norm"]):
            return False
    best_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(str(summary["checkpoint_path"])), best_root / "model.pt")
    best_summary = dict(summary)
    best_summary["checkpoint_path"] = str(best_root / "model.pt")
    write_json(best_summary_path, best_summary)
    return True


def main() -> dict:
    general_config = prepare.load_train_config()
    prepare.set_global_seed(general_config["seed"])
    window_rng = np.random.default_rng(general_config["seed"])

    validation_raw = prepare.load_validation_sequence()
    history_window = int(general_config["history_window"])
    window_min_length = int(general_config["window_min_length"])
    train_split_index = int(general_config["train_split_index"])
    full_window_length = train_split_index - history_window

    normalizer = prepare.Normalizer.fit([validation_raw], history_window=history_window)
    validation_norm = normalizer.normalize_sequence(validation_raw)

    u_full, y_full, sampling_time = prepare.load_raw_train_arrays()
    crops_per_step = max(1, int(config_pars.get("crops_per_step", 1)))
    extend_windows_to_range_end = bool(config_pars.get("extend_windows_to_range_end", False))
    long_window_prob = float(config_pars.get("long_window_prob", 0.0))

    def sample_fresh_training_windows():
        windows = []
        for _ in range(crops_per_step):
            use_long = long_window_prob > 0.0 and window_rng.random() < long_window_prob
            window = prepare.sample_training_window(
                u_full=u_full, y_full=y_full, sampling_time=sampling_time, history_window=history_window,
                max_index=train_split_index,
                min_window_length=full_window_length if use_long else window_min_length,
                rng=window_rng,
                extend_to_range_end=True if use_long else extend_windows_to_range_end,
            )
            windows.append(normalizer.normalize_sequence(window))
        return windows

    device = general_config["device"]
    checkpoint_root = Path(general_config["checkpoint_path"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_dir = Path(general_config["log_dir"])
    plots_dir = Path(general_config["plots_path"])

    model = build_model(config_pars, general_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config_pars["lr"], weight_decay=config_pars.get("weight_decay", 0.0))

    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    start_time = time.perf_counter()

    _time_budget_override = config_pars.get("time_budget_seconds")
    time_budget_seconds = float(_time_budget_override if _time_budget_override is not None else general_config["time_budget_seconds"])
    eval_every = int(general_config["eval_every"])
    max_epochs = int(config_pars["max_epochs"])
    drift_penalty_weight = float(config_pars.get("drift_penalty_weight", 0.0))

    log_path = log_dir / "train.log"
    initialize_log(log_path)

    initial_validation_metrics = evaluate_sequence(model, validation_raw, validation_norm, normalizer, device)
    with torch.no_grad():
        initial_loss = compute_training_loss(model, sample_fresh_training_windows(), device=device, drift_penalty_weight=drift_penalty_weight)
    best_val_rmse_norm = float(initial_validation_metrics["rmse_norm"])
    initial_line = f"0,{initial_loss.item():.6f},{initial_validation_metrics['rmse_norm']:.6f}"
    print(initial_line)
    append_log_line(log_path, initial_line)

    skipped_nan_steps = 0
    hit_time_budget = False
    for epoch in range(1, max_epochs + 1):
        if time.perf_counter() - start_time >= time_budget_seconds:
            hit_time_budget = True
            break

        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        window_list_norm = sample_fresh_training_windows()
        loss = compute_training_loss(model, window_list_norm, device=device, drift_penalty_weight=drift_penalty_weight)
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

        if epoch == 1:
            warn_if_time_budget_too_tight(
                seconds_per_epoch=time.perf_counter() - epoch_start, max_epochs=max_epochs,
                time_budget_seconds=time_budget_seconds, model_type=str(config_pars.get("type", "")).upper(),
            )

        should_evaluate = epoch == 1 or epoch % eval_every == 0 or epoch == max_epochs
        if not should_evaluate:
            continue
        if time.perf_counter() - start_time >= time_budget_seconds:
            hit_time_budget = True
            break

        validation_metrics = evaluate_sequence(model, validation_raw, validation_norm, normalizer, device)
        line = f"{epoch},{loss.item():.6f},{validation_metrics['rmse_norm']:.6f}"
        print(line)
        append_log_line(log_path, line)

        # Bookkeeping only -- NOT early stopping. The loop never breaks because of
        # this comparison, only because of max_epochs or time_budget_seconds above.
        if validation_metrics["rmse_norm"] < best_val_rmse_norm:
            best_val_rmse_norm = float(validation_metrics["rmse_norm"])
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

    if hit_time_budget:
        print(f"[train] stopped by time_budget_seconds={time_budget_seconds:.1f}s safety net (not early stopping)")

    model.load_state_dict(best_state_dict)
    final_validation_metrics = evaluate_sequence(model, validation_raw, validation_norm, normalizer, device)

    checkpoint_payload = {
        "model_state_dict": model.state_dict(), "model_config": copy.deepcopy(config_pars),
        "general_config": general_config, "normalizer": normalizer.state_dict(), "best_epoch": best_epoch,
        "hit_time_budget": hit_time_budget,
        "metrics": {
            "validation_rmse_norm": float(final_validation_metrics["rmse_norm"]),
            "validation_rmse_raw": float(final_validation_metrics["rmse_raw"]),
        },
    }
    torch.save(checkpoint_payload, checkpoint_root / "model.pt")

    summary = {
        "metric": "rmse", "validation_rmse_norm": float(final_validation_metrics["rmse_norm"]),
        "validation_rmse_raw": float(final_validation_metrics["rmse_raw"]), "best_epoch": best_epoch,
        "hit_time_budget": hit_time_budget,
        "checkpoint_path": str(checkpoint_root / "model.pt"), "log_path": str(log_path),
        "skipped_nan_steps": skipped_nan_steps,
    }
    write_json(checkpoint_root / "summary.json", summary)

    best_so_far_updated = maybe_update_best_so_far(checkpoint_root, summary)
    plot_training_curve(log_path, plots_dir / "training_curve.png")
    plot_fit_diagnostic(model, validation_raw, validation_norm, normalizer, device, plots_dir / "train_fit.png")

    print("")
    print("Training summary")
    print(f"Validation RMSE norm : {summary['validation_rmse_norm']:.6f}")
    print(f"Best epoch           : {best_epoch}")
    print(f"Hit time budget      : {hit_time_budget}")
    print(f"Best-so-far checkpoint: {'updated' if best_so_far_updated else 'kept previous'}")
    print(f"Diagnostics saved to  : {plots_dir / 'training_curve.png'}, {plots_dir / 'train_fit.png'}")
    if skipped_nan_steps > 0:
        print(f"NOTE: {skipped_nan_steps} step(s) had a non-finite gradient and were skipped.")
    return summary


if __name__ == "__main__":
    main()