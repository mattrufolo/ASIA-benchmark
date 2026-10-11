from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import prepare
from model import build_model_from_config


def load_checkpoint_paths(checkpoint_root: Path, checkpoint_set: str) -> list[Path]:
    root = checkpoint_root / checkpoint_set if checkpoint_set == "best_so_far" else checkpoint_root
    paths = sorted(p for p in root.glob("*/model.pt") if p.parent.name != "best_so_far")
    if not paths:
        raise FileNotFoundError(f"No fold checkpoints found under {root}. Run train.py first.")
    return paths


def build_model(model_config, general_config):
    return build_model_from_config(
        config_pars=model_config, n_inputs=general_config["n_inputs"],
        n_states=general_config["n_states"], n_outputs=general_config["n_outputs"],
    ).to(general_config["device"])


def predict_with_ensemble(checkpoints, sequence_raw, general_config):
    device = general_config["device"]
    predictions_raw = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = build_model(checkpoint["model_config"], general_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        normalizer = prepare.Normalizer.from_state_dict(checkpoint["normalizer"])
        sequence_norm = normalizer.normalize_sequence(sequence_raw)
        with torch.no_grad():
            y_hat_norm, _ = model(sequence_norm.u.to(device), sequence_norm.y0.to(device))
            y_hat_raw = normalizer.denormalize_y_tensor(y_hat_norm)
        predictions_raw.append(y_hat_raw.detach().cpu().numpy()[0])
    return np.mean(np.stack(predictions_raw, axis=0), axis=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fold-checkpoint ensemble on the official F16 test datasets.")
    parser.add_argument("--checkpoint-set", choices=["current", "best_so_far"], default="current")
    args = parser.parse_args()

    _, test_sequences_full, general_config = prepare.load_datasets_and_config()

    checkpoint_root = Path(general_config["checkpoint_path"])
    checkpoints = load_checkpoint_paths(checkpoint_root, args.checkpoint_set)
    print(f"Evaluating ensemble of {len(checkpoints)} fold checkpoints ({args.checkpoint_set}) on {len(test_sequences_full)} official test datasets.")

    results = {}
    for name, sequence in sorted(test_sequences_full.items()):
        y_hat_raw = predict_with_ensemble(checkpoints, sequence, general_config)
        y_true_raw = sequence.y[0].detach().cpu().numpy()
        per_output_rmse = [
            prepare.rmse(y_true_raw[sequence.warmup:, ch], y_hat_raw[sequence.warmup:, ch])
            for ch in range(y_true_raw.shape[-1])
        ]
        # Official submission format (confirmed directly against
        # nonlinear_benchmarks' own submission_examples/F16.py this
        # session): ONE combined RMSE per test set, pooling all 3 output
        # channels together via RMSE(test.y[n:], prediction[n:]) on the
        # full (T, 3) arrays -- NOT averaged per-channel RMSEs. This is
        # what produces the standard "6 numbers" format matching the
        # public leaderboard (nonlinearbenchmark.org/benchmarks/f-16-gvt).
        combined_rmse = prepare.rmse(y_true_raw[sequence.warmup:], y_hat_raw[sequence.warmup:])
        results[name] = {
            "rmse_per_output": per_output_rmse,
            "rmse_mean_over_outputs": float(np.mean(per_output_rmse)),
            "rmse_combined_official_format": combined_rmse,
        }

    official_rmse_list = [results[name]["rmse_combined_official_format"] for name in sorted(results)]
    overall_mean = float(np.mean([r["rmse_mean_over_outputs"] for r in results.values()]))

    summary = {
        "units": "m/s^2", "checkpoint_set": args.checkpoint_set, "num_checkpoints": len(checkpoints),
        "output_names": prepare.output_names, "per_dataset": results, "overall_mean_rmse": overall_mean,
        "official_format_rmse_list": official_rmse_list,
        "official_format_dataset_order": sorted(results.keys()),
    }
    (checkpoint_root / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    print(f"{'dataset':<24} " + " ".join(f"{name:>22}" for name in prepare.output_names) + "   combined")
    for name, r in sorted(results.items()):
        vals = " ".join(f"{v:22.6f}" for v in r["rmse_per_output"])
        print(f"{name:<24} {vals}   {r['rmse_combined_official_format']:.6f}")
    print("")
    print("Official submission format (nonlinear_benchmarks' own F16.py: one combined RMSE per test")
    print(f"set, pooling all 3 outputs, dataset order {sorted(results.keys())}):")
    print("RMSE to submit = [", *(f"{x:.4f}" for x in official_rmse_list), "] m/s^2")
    print("")
    print(f"Overall mean RMSE (per-channel breakdown, averaged) = {overall_mean:.6f} m/s^2")
    print(f"Saved metrics: {checkpoint_root / 'test_metrics.json'}")
    print("Monitoring only -- never use this to choose between candidates. See program.md.")
    return summary


if __name__ == "__main__":
    main()