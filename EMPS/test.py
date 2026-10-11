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


def load_checkpoint_path(checkpoint_root: Path, checkpoint_set: str) -> Path:
    if checkpoint_set == "current":
        path = checkpoint_root / "model.pt"
    else:
        path = checkpoint_root / "best_so_far" / "model.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}. Run `train.py` first.")
    return path


def load_model_and_normalizer(
    checkpoint_path: Path,
    general_config: dict,
) -> tuple[torch.nn.Module, prepare.Normalizer]:
    # These checkpoints are produced locally by train.py and contain pickled numpy
    # state in addition to tensors, so we load them in trusted mode.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=general_config["device"],
        weights_only=False,
    )
    model = build_model_from_config(
        config_pars=checkpoint["model_config"],
        n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"],
        n_outputs=general_config["n_outputs"],
        dt=float(general_config.get("sampling_time", 0.001)),
    ).to(general_config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = prepare.Normalizer.from_state_dict(checkpoint["normalizer"])
    return model, normalizer


def predict_with_checkpoint(
    checkpoint_path: Path,
    test_sequence: prepare.EMPSSequence,
    general_config: dict,
) -> np.ndarray:
    model, normalizer = load_model_and_normalizer(checkpoint_path, general_config)
    test_sequence_norm = normalizer.normalize_sequence(test_sequence)

    with torch.no_grad():
        y_hat_norm, _ = model(
            test_sequence_norm.u.to(general_config["device"]),
            test_sequence_norm.y0.to(general_config["device"]),
        )
        y_hat = normalizer.denormalize_y_tensor(y_hat_norm).detach().cpu().numpy()

    # Test-time predictions are returned in physical units (meters).
    return y_hat[0]


def plot_test_predictions(
    test_sequence: prepare.EMPSSequence,
    prediction: np.ndarray,
    plot_path: Path,
) -> None:
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    time_axis = np.arange(test_sequence.num_samples, dtype=np.float32) * test_sequence.sampling_time
    u_values = test_sequence.u[0].detach().cpu().numpy()
    y_true = test_sequence.y[0].detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(time_axis, u_values[:, 0], linewidth=1.5, color="#006d77")
    axes[0].set_ylabel(prepare.input_names[0])
    axes[0].set_title("Test input trajectory (motor force, N)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, y_true[:, 0], linewidth=2.0, color="#1d3557", label="true position")
    axes[1].plot(time_axis, prediction[:, 0], linewidth=1.3, color="#e76f51", label="model prediction")
    axes[1].axvline(
        test_sequence.warmup * test_sequence.sampling_time,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="test warmup",
    )
    axes[1].set_ylabel(prepare.output_names[0] + " [m]")
    axes[1].set_xlabel("time [s]")
    axes[1].set_title("Test output prediction (denormalized, physical units)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved checkpoint on the official EMPS test set.")
    parser.add_argument(
        "--checkpoint-set",
        choices=("current", "best_so_far"),
        default="current",
        help="Which saved checkpoint to evaluate.",
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    _, test_sequence, general_config = prepare.load_datasets_and_config()

    checkpoint_root = Path(general_config["checkpoint_path"])
    plots_root = Path(general_config["plots_path"])

    checkpoint_path = load_checkpoint_path(checkpoint_root, args.checkpoint_set)
    prediction = predict_with_checkpoint(checkpoint_path, test_sequence=test_sequence, general_config=general_config)

    y_true = test_sequence.y[0].detach().cpu().numpy()
    test_rmse_m = prepare.rmse(y_true[test_sequence.warmup :], prediction[test_sequence.warmup :])
    # The official EMPS benchmark (nonlinear_benchmarks) reports RMSE in millimeters.
    test_rmse_mm = 1000.0 * test_rmse_m

    metrics = {
        "metric": "rmse",
        "metric_scale": "denormalized_physical_units_meters",
        "plot_scale": "denormalized_physical_units_meters",
        "warmup_test": int(test_sequence.warmup),
        "test_rmse_m": float(test_rmse_m),
        "test_rmse_mm": float(test_rmse_mm),
        "checkpoint_path": str(checkpoint_path),
    }

    output_checkpoint_root = checkpoint_root if args.checkpoint_set == "current" else checkpoint_root / "best_so_far"
    output_plot_root = plots_root if args.checkpoint_set == "current" else plots_root / "best_so_far"

    output_checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_plot_root.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_checkpoint_root / "test_predictions.npz",
        y_true=y_true.astype(np.float32),
        prediction=prediction.astype(np.float32),
    )
    (output_checkpoint_root / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    plot_test_predictions(
        test_sequence=test_sequence,
        prediction=prediction,
        plot_path=output_plot_root / "test_prediction.png",
    )

    print(f"Test summary ({args.checkpoint_set})")
    print(f"Test RMSE : {metrics['test_rmse_mm']:.4f} mm ({metrics['test_rmse_m']:.6f} m)")
    print(f"Saved metrics : {output_checkpoint_root / 'test_metrics.json'}")

    return metrics


if __name__ == "__main__":
    main()
