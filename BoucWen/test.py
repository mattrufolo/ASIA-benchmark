from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import prepare
from model import build_model_from_config


def load_checkpoint_paths(checkpoint_root: Path, checkpoint_set: str) -> list[Path]:
    summary_root = checkpoint_root if checkpoint_set == "current" else checkpoint_root / "best_so_far"
    summary_path = summary_root / "cross_validation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return [Path(item["checkpoint_path"]) for item in summary["folds"]]
    best_summary_path = checkpoint_root / "best_so_far" / "cross_validation_summary.json"
    if best_summary_path.exists():
        summary = json.loads(best_summary_path.read_text(encoding="utf-8"))
        return [Path(item["checkpoint_path"]) for item in summary["folds"]]
    return sorted(checkpoint_root.glob("fold_*/model.pt"))


def load_model_and_normalizer(checkpoint_path: Path, general_config: dict):
    checkpoint = torch.load(checkpoint_path, map_location=general_config["device"], weights_only=False)
    model = build_model_from_config(
        config_pars=checkpoint["model_config"], n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"], n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    normalizer = prepare.Normalizer.from_state_dict(checkpoint["normalizer"])
    return model, normalizer


def model_requires_raw_io(model) -> bool:
    """See train.py's identical function for the full explanation --
    models whose parameters are calibrated to real physical units (e.g.
    BoucWenModel) need RAW, un-normalized u/y0 as input."""
    return bool(getattr(model, "requires_raw_io", False))


def call_model(model, sequence_norm, sequence_raw, normalizer, device, teacher_forcing_y=None):
    kwargs = {}
    if hasattr(model, "num_substeps"):
        kwargs["sampling_time"] = sequence_raw.sampling_time
    elif getattr(model, "supports_prediction_mode", False):
        kwargs["teacher_forcing_y"] = teacher_forcing_y.to(device) if teacher_forcing_y is not None else None

    if model_requires_raw_io(model):
        y_hat_raw, hidden_state = model(sequence_raw.u.to(device), sequence_raw.y0.to(device), **kwargs)
    else:
        y_hat_norm, hidden_state = model(sequence_norm.u.to(device), sequence_norm.y0.to(device), **kwargs)
        y_hat_raw = normalizer.denormalize_y_tensor(y_hat_norm)
    return y_hat_raw, hidden_state


def predict_with_checkpoint(checkpoint_path: Path, test_sequence, general_config: dict, mode: str = "simulation") -> np.ndarray:
    model, normalizer = load_model_and_normalizer(checkpoint_path, general_config)
    test_sequence_norm = normalizer.normalize_sequence(test_sequence)
    teacher_forcing_y = None
    if mode == "prediction" and getattr(model, "supports_prediction_mode", False):
        # Shifted-by-one true output (normalized), matching the paper's own
        # prediction-mode definition: y_mod(t) = F(u(1..t), y(1..t-1)).
        y_true_norm = test_sequence_norm.y
        teacher_forcing_y = torch.cat([torch.zeros_like(y_true_norm[:, :1, :]), y_true_norm[:, :-1, :]], dim=1)
    # cuDNN's RNN kernel throws CUDNN_STATUS_NOT_SUPPORTED on the very long
    # (153000-sample) sine-sweep sequence on some GPU/cuDNN combinations;
    # disabling cudnn for this single-shot, monitoring-only inference call
    # (correctness matters here, not speed) sidesteps it without touching
    # the evaluation protocol itself.
    with torch.no_grad(), torch.backends.cudnn.flags(enabled=False):
        y_hat_raw, _ = call_model(model, test_sequence_norm, test_sequence, normalizer, general_config["device"], teacher_forcing_y=teacher_forcing_y)
    return y_hat_raw.detach().cpu().numpy()[0]


def model_supports_prediction_mode(checkpoint_path: Path, general_config: dict) -> bool:
    model, _ = load_model_and_normalizer(checkpoint_path, general_config)
    return bool(getattr(model, "supports_prediction_mode", False))


def plot_test_predictions(test_sequence, fold_predictions, ensemble_prediction, plot_path: Path) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(test_sequence.num_samples, dtype=np.float32) * test_sequence.sampling_time
    u_values = test_sequence.u[0].detach().cpu().numpy()
    y_true = test_sequence.y[0].detach().cpu().numpy()
    ensemble = ensemble_prediction

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(time_axis, u_values[:, 0], linewidth=0.4, color="#006d77")
    axes[0].set_ylabel(prepare.input_names[0] + " [N]")
    axes[0].set_title(f"Test input trajectory ({test_sequence.name})")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, y_true[:, 0], linewidth=0.6, color="#1d3557", label="true output")
    for index, prediction in enumerate(fold_predictions, start=1):
        axes[1].plot(time_axis, prediction[:, 0], linewidth=0.5, alpha=0.3, color="#6c757d",
                     label="individual fold models" if index == 1 else None)
    axes[1].plot(time_axis, ensemble[:, 0], linewidth=0.8, color="#e76f51", label="ensemble mean")
    axes[1].set_ylabel(prepare.output_names[0] + " [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title(f"Test output prediction ({test_sequence.name}, denormalized, simulation mode)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the fold-checkpoint ensemble on BOTH official, real test "
            "signals (test_multisine, test_sinesweep), reported SEPARATELY. "
            "Reports simulation-mode RMSE always, and ALSO prediction-mode "
            "RMSE if the checkpointed model supports output feedback "
            "(use_output_feedback=True) -- matching the benchmark's own "
            "request (Noel & Schoukens 2016, Section 5) to report both "
            "figures of merit where the model architecture allows it."
        )
    )
    parser.add_argument("--checkpoint-set", choices=("current", "best_so_far"), default="current")
    return parser.parse_args()


def main() -> dict:
    args = parse_args()
    _, test_sequences, general_config = prepare.load_datasets_and_config()

    checkpoint_root = Path(general_config["checkpoint_path"])
    plots_root = Path(general_config["plots_path"])
    checkpoint_paths = load_checkpoint_paths(checkpoint_root, args.checkpoint_set)
    if not checkpoint_paths:
        raise FileNotFoundError("No fold checkpoints found. Run `train.py` first.")

    output_checkpoint_root = checkpoint_root if args.checkpoint_set == "current" else checkpoint_root / "best_so_far"
    output_plot_root = plots_root if args.checkpoint_set == "current" else plots_root / "best_so_far"
    output_checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_plot_root.mkdir(parents=True, exist_ok=True)

    supports_prediction = model_supports_prediction_mode(checkpoint_paths[0], general_config)

    per_test_metrics: dict[str, dict] = {}

    for test_name in sorted(test_sequences.keys()):
        test_sequence = test_sequences[test_name]

        fold_predictions_sim = [predict_with_checkpoint(path, test_sequence, general_config, mode="simulation") for path in checkpoint_paths]
        ensemble_sim = np.mean(np.stack(fold_predictions_sim, axis=0), axis=0)

        y_true = test_sequence.y[0].detach().cpu().numpy()
        ensemble_rmse_sim = prepare.rmse(y_true[test_sequence.warmup:], ensemble_sim[test_sequence.warmup:])
        fold_rmses_sim = [prepare.rmse(y_true[test_sequence.warmup:], p[test_sequence.warmup:]) for p in fold_predictions_sim]

        entry = {
            "fold_test_rmse_simulation": [float(v) for v in fold_rmses_sim],
            "ensemble_test_rmse_simulation": float(ensemble_rmse_sim),
        }

        if supports_prediction:
            fold_predictions_pred = [predict_with_checkpoint(path, test_sequence, general_config, mode="prediction") for path in checkpoint_paths]
            ensemble_pred = np.mean(np.stack(fold_predictions_pred, axis=0), axis=0)
            ensemble_rmse_pred = prepare.rmse(y_true[test_sequence.warmup:], ensemble_pred[test_sequence.warmup:])
            fold_rmses_pred = [prepare.rmse(y_true[test_sequence.warmup:], p[test_sequence.warmup:]) for p in fold_predictions_pred]
            entry["fold_test_rmse_prediction"] = [float(v) for v in fold_rmses_pred]
            entry["ensemble_test_rmse_prediction"] = float(ensemble_rmse_pred)

        per_test_metrics[test_name] = entry
        np.savez_compressed(
            output_checkpoint_root / f"{test_name}_ensemble_predictions.npz",
            y_true=y_true.astype(np.float32), ensemble_prediction=ensemble_sim.astype(np.float32),
            fold_predictions=np.stack(fold_predictions_sim, axis=0).astype(np.float32),
        )
        plot_test_predictions(test_sequence, fold_predictions_sim, ensemble_sim, output_plot_root / f"{test_name}_ensemble_prediction.png")

    metrics = {
        "metric": "rmse", "metric_scale": "meters", "num_models": len(checkpoint_paths),
        "supports_prediction_mode": supports_prediction, "per_test": per_test_metrics,
        "note": "test_multisine and test_sinesweep are REAL official data, reported SEPARATELY, never combined.",
    }
    (output_checkpoint_root / "test_ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Test ensemble summary ({args.checkpoint_set})")
    print(f"Number of fold models: {len(checkpoint_paths)}")
    print(f"Model supports prediction mode: {supports_prediction}")
    for test_name, values in sorted(per_test_metrics.items()):
        line = f"  {test_name:16s}: simulation RMSE = {values['ensemble_test_rmse_simulation']:.6e} m"
        if "ensemble_test_rmse_prediction" in values:
            line += f"  |  prediction RMSE = {values['ensemble_test_rmse_prediction']:.6e} m"
        print(line)
    print(f"Saved metrics: {output_checkpoint_root / 'test_ensemble_metrics.json'}")
    return metrics


if __name__ == "__main__":
    main()