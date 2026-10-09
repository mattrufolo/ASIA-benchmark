from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import prepare
from model import build_model_from_config


def load_checkpoint_paths(checkpoint_root: Path, checkpoint_set: str) -> list[Path]:
    root = checkpoint_root / checkpoint_set if checkpoint_set == "best_so_far" else checkpoint_root
    paths = sorted(p for p in root.glob("fold_*/model.pt"))
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


def group_by_level(test_sequences: dict) -> dict[str, list[str]]:
    """test_{levelNN}_r{RR} -> groups by levelNN, matching the 3
    amplitude levels (100/200/300 mV RMS -- see FSM_description.md)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for name in test_sequences:
        match = re.match(r"test_(level\d+)_r\d+", name)
        level = match.group(1) if match else "unknown"
        groups[level].append(name)
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fold-checkpoint ensemble on the official FSM test datasets.")
    parser.add_argument("--checkpoint-set", choices=["current", "best_so_far"], default="best_so_far")
    args = parser.parse_args()

    _, test_sequences, general_config = prepare.load_datasets_and_config()
    checkpoint_root = Path(general_config["checkpoint_path"])
    checkpoints = load_checkpoint_paths(checkpoint_root, args.checkpoint_set)
    print(f"Evaluating ensemble of {len(checkpoints)} fold checkpoints ({args.checkpoint_set}) on {len(test_sequences)} official test sequences.")

    level_groups = group_by_level(test_sequences)
    output_names = prepare.output_names

    summary = {"units": "micrometers", "checkpoint_set": args.checkpoint_set, "num_checkpoints": len(checkpoints),
               "output_names": output_names, "per_level": {}}

    for level, seq_names in sorted(level_groups.items()):
        # Per-output RMSE, pooled (concatenated) across all realizations
        # in this level -- matches FineSteeringMirror.py's own "averaged
        # over realizations and periods" convention (our periods are
        # already concatenated into each sequence at prepare.py time, so
        # pooling across realizations here completes the same averaging).
        per_output_sq_errors = [[] for _ in range(len(output_names))]
        for seq_name in sorted(seq_names):
            sequence_raw = test_sequences[seq_name]
            y_hat_raw = predict_with_ensemble(checkpoints, sequence_raw, general_config)
            y_true_raw = sequence_raw.y[0].detach().cpu().numpy()
            for ch in range(len(output_names)):
                diff = y_true_raw[sequence_raw.warmup:, ch] - y_hat_raw[sequence_raw.warmup:, ch]
                per_output_sq_errors[ch].append(diff ** 2)

        per_output_rmse_um = []
        per_output_nrmse_pct = []
        for ch in range(len(output_names)):
            pooled_sq_error = np.concatenate(per_output_sq_errors[ch])
            rmse_m = float(np.sqrt(np.mean(pooled_sq_error)))
            rmse_um = rmse_m * 1e6  # meters -> micrometers, matching the official submission format
            true_all = np.concatenate([
                test_sequences[n].y[0].detach().cpu().numpy()[test_sequences[n].warmup:, ch] for n in sorted(seq_names)
            ])
            std_true = float(np.std(true_all)) + 1e-12
            nrmse_pct = 100.0 * (rmse_m / std_true)
            per_output_rmse_um.append(rmse_um)
            per_output_nrmse_pct.append(nrmse_pct)

        combined_rmse_um = float(np.mean(per_output_rmse_um))  # matches "RMSE to submit" = mean across the 3 outputs
        summary["per_level"][level] = {
            "rmse_per_output_um": per_output_rmse_um, "nrmse_per_output_pct": per_output_nrmse_pct,
            "rmse_combined_um": combined_rmse_um,
        }

    overall_mean_um = float(np.mean([v["rmse_combined_um"] for v in summary["per_level"].values()]))
    summary["overall_mean_rmse_um"] = overall_mean_um
    (checkpoint_root / "test_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("")
    for level, values in sorted(summary["per_level"].items()):
        print(f"{level}:")
        print(f"  RMSE to submit: {values['rmse_combined_um']:.3e} \u00b5m")
        for i, name in enumerate(output_names):
            print(f"  {name}: {values['rmse_per_output_um'][i]:.3e} \u00b5m ({values['nrmse_per_output_pct'][i]:.2f}%)")
    print("")
    print(f"Overall mean RMSE (across 3 amplitude levels) = {overall_mean_um:.3e} \u00b5m")
    print(f"Saved metrics: {checkpoint_root / 'test_metrics.json'}")
    print("Monitoring only -- never use this to choose between candidates. See program.md.")
    return summary


if __name__ == "__main__":
    main()