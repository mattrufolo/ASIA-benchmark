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


def default_checkpoint_path(checkpoint_root: Path) -> Path:
    final_path = checkpoint_root / "final" / "model.pt"
    if final_path.exists():
        return final_path

    search_path = checkpoint_root / "search" / "model.pt"
    if search_path.exists():
        print(
            "WARNING: no refit checkpoint found (checkpoints/final/model.pt). Falling back to "
            "the SEARCH checkpoint (checkpoints/search/model.pt), which was only trained on "
            "[history_window:search_train_stop]. Run `refit.py` for the official test number."
        )
        return search_path

    raise FileNotFoundError(
        "No checkpoint found under checkpoints/final or checkpoints/search. "
        "Run `train.py` (and then `refit.py`) first."
    )


def load_model_and_normalizer(
    checkpoint_path: Path,
    general_config: dict,
) -> tuple[torch.nn.Module, prepare.Normalizer, dict]:
    # These checkpoints are produced locally by train.py/refit.py and contain
    # pickled numpy state in addition to tensors, so we load them in trusted mode.
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
    ).to(general_config["device"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    normalizer = prepare.Normalizer.from_state_dict(checkpoint["normalizer"])
    return model, normalizer, checkpoint


def predict_sequence(
    model: torch.nn.Module,
    test_sequence: prepare.CEDSequence,
    normalizer: prepare.Normalizer,
    general_config: dict,
) -> np.ndarray:
    test_sequence_norm = normalizer.normalize_sequence(test_sequence)
    with torch.no_grad():
        y_hat_norm, _ = model(
            test_sequence_norm.u.to(general_config["device"]),
            test_sequence_norm.y0.to(general_config["device"]),
        )
        y_hat = normalizer.denormalize_y_tensor(y_hat_norm).detach().cpu().numpy()
    # Predictions are returned in physical units (ticks/s).
    return y_hat[0]


def plot_test_prediction(
    test_sequence: prepare.CEDSequence,
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
    axes[0].set_title(f"Test input trajectory ({test_sequence.name})")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, y_true[:, 0], linewidth=2.0, color="#1d3557", label="true output")
    axes[1].plot(time_axis, prediction[:, 0], linewidth=2.0, color="#e76f51", label="model prediction")
    axes[1].axvline(
        test_sequence.warmup * test_sequence.sampling_time,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="test warmup",
    )
    axes[1].set_ylabel(prepare.output_names[0])
    axes[1].set_xlabel("time [s]")
    axes[1].set_title(f"Test output prediction ({test_sequence.name}, denormalized)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3 of the CED procedure: evaluate ONE model checkpoint on the official "
            "[400:500] test data. Defaults to checkpoints/final/model.pt, produced by "
            "refit.py. This number is for reporting only and must never be used to "
            "reconsider the architecture/hyperparameter choice made using train.py's "
            "search-validation RMSE."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Explicit checkpoint file to evaluate. Defaults to checkpoints/final/model.pt "
            "(falling back to checkpoints/search/model.pt with a warning, for monitoring "
            "purposes only, if no refit checkpoint exists yet)."
        ),
    )
    return parser.parse_args()


def main() -> dict[str, object]:
    args = parse_args()
    _, test_sequences, general_config = prepare.load_refit_datasets_and_config()

    checkpoint_root = Path(general_config["checkpoint_path"])
    plots_root = Path(general_config["plots_path"])

    checkpoint_path = args.checkpoint if args.checkpoint is not None else default_checkpoint_path(checkpoint_root)
    model, normalizer, checkpoint = load_model_and_normalizer(checkpoint_path, general_config)
    stage = checkpoint.get("stage", "unknown")

    # Explicit low->high order, matching the official submission_examples/CED.py
    # convention ("RMSE to submit = [test_1_RMSE; test_2_RMSE]"), rather than
    # relying on alphabetical dict order (which would put "high" before "low").
    ordered_test_names = [f"test_{realization}" for realization in prepare.REALIZATIONS]
    ordered_test_items = [(name, test_sequences[name]) for name in ordered_test_names if name in test_sequences]

    per_sequence_metrics: dict[str, float] = {}
    submission_rmse: list[float] = []

    output_plot_root = plots_root / stage
    output_plot_root.mkdir(parents=True, exist_ok=True)

    for test_name, test_sequence in ordered_test_items:
        prediction = predict_sequence(
            model=model,
            test_sequence=test_sequence,
            normalizer=normalizer,
            general_config=general_config,
        )
        y_true = test_sequence.y[0].detach().cpu().numpy()
        test_rmse = prepare.rmse(y_true[test_sequence.warmup :], prediction[test_sequence.warmup :])

        per_sequence_metrics[test_name] = float(test_rmse)
        submission_rmse.append(float(test_rmse))

        np.savez_compressed(
            output_plot_root / f"{test_name}_prediction.npz",
            y_true=y_true.astype(np.float32),
            prediction=prediction.astype(np.float32),
        )
        plot_test_prediction(
            test_sequence=test_sequence,
            prediction=prediction,
            plot_path=output_plot_root / f"{test_name}_prediction.png",
        )

    metrics = {
        "metric": "rmse",
        "metric_scale": "denormalized_physical_units_ticks_per_s",
        "checkpoint_stage": stage,
        "checkpoint_path": str(checkpoint_path),
        "per_sequence": per_sequence_metrics,
        # Matches the official submission_examples/CED.py reporting convention:
        # "RMSE to submit = [test_1_RMSE; test_2_RMSE]"
        "submission_rmse_low_high": submission_rmse,
        "submission_rmse_mean": float(np.mean(submission_rmse)),
    }

    (output_plot_root / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Test summary (checkpoint stage: {stage})")
    print(f"Checkpoint: {checkpoint_path}")
    for test_name, value in per_sequence_metrics.items():
        print(f"{test_name:24s}: RMSE = {value:.4f} ticks/s")
    order_label = "; ".join(prepare.REALIZATIONS)
    print(f"Submission RMSE [{order_label}] : {[f'{v:.4f}' for v in submission_rmse]}")
    print(f"Saved metrics: {output_plot_root / 'test_metrics.json'}")
    if stage != "final_refit":
        print(
            "NOTE: this is not the official reporting number -- it was evaluated on a "
            "search-stage checkpoint. Run `refit.py` first for the number that should "
            "actually be reported/submitted."
        )

    return metrics


if __name__ == "__main__":
    main()