# Wiener-Hammerstein Benchmark With Process Noise Program

This folder contains the material used to run autoresearch on the Wiener-Hammerstein-with-process-noise benchmark. The goal is to train and evaluate models that describe the output voltage from the input voltage, under a system whose DOMINANT noise source is process noise entering before the static nonlinearity (not just measurement noise).

## IMPORTANT: Read This Before Starting

**Minimum 15 total iterations before concluding the search.** This is a hard floor, not a soft target -- do not stop early just because a result looks conclusive.

**Data setup (required before running `prepare.py`)**: this benchmark's data is not reliably available through the `nonlinear_benchmarks` package. Copy the official zip's extracted `WienerHammersteinFiles/` folder (from data.4tu.nl, article 12952124) directly into this project's own directory, so `WienerHammersteinFiles/WH_EstimationExample.mat` and `WienerHammersteinFiles/WH_TestDataset.mat` exist alongside `prepare.py`. See `prepare.py`'s own `RAW_MAT_DIR` comment for the exact expected layout.

The fold-design question has already been settled: the default 5-fold design (10 realizations from WH_EstimationExample.mat) was directly compared against alternatives and confirmed the better choice from the cross-validation curves and test-signal fits. This is committed going forward -- do not re-run this comparison as one of the 15 required iterations. See the fallback below if the design underperforms unexpectedly during the search.

```
step 1: SEARCH   train.py   -- leave-one-fold-out CV over 5 folds of
                                2 realizations each (10 total, from
                                WH_EstimationExample.mat)
step 2: TEST     test.py    -- evaluate the ENSEMBLE of the 5 fold checkpoints
                                on BOTH official noise-free test signals
                                (multisine, sine-sweep), reported separately
```

Start from first principles, not assumed conclusions. No model family should be assumed to win or lose ahead of time. Reason from WienerHammersteinProcessNoise_description.md's physical description, then let validation_rmse_norm_mean decide.

## Benchmark Description

Read WienerHammersteinProcessNoise_description.md before starting: it covers the process-noise structure, the two candidate estimation datasets and why one was chosen as the default, and the fold-design rationale.

## Folder Structure

```
prepare.py
model.py
train.py
test.py
WienerHammersteinProcessNoise_description.md
WienerHammersteinFiles/    <- you place this (see Data setup above)
```

## `prepare.py`

Reads the official data DIRECTLY from a local `WienerHammersteinFiles/` folder (place the official zip's extracted contents there -- see `prepare.py`'s own `RAW_MAT_DIR` comment for the exact expected layout; data.4tu.nl, article 12952124). No `nonlinear_benchmarks` package dependency for this project -- that benchmark isn't reliably available through it, so this reads the `.mat` files directly via `scipy.io`. Reads two specific files by name: `WH_EstimationExample.mat` (10 realizations, grouped into 5 folds of 2) and `WH_TestDataset.mat` (2 noise-free test signals). Builds initial-condition vectors (`history_window = 20`, matching a third-party paper's own convention for this exact benchmark). Read-only.

## `model.py`

Contains a black-box recurrent baseline (RNN/GRU/LSTM) and WienerHammersteinProcessNoiseModel (type: "WIENERHAMMERSTEIN"): the true R -> f -> S structure, each LTI block a learnable FIR filter. See WienerHammersteinProcessNoise_description.md Section 5 for why FIR (not a recursive state-space realization) was chosen, and for a genuinely different avenue worth exploring (explicit process-noise modeling). Can be modified.

## `train.py`

Leave-one-fold-out CV, 5 parallel workers. config_pars holds model/ training hyperparameters (can be modified), including optional per-experiment fold_time_budget_seconds/eval_every overrides (default None -> use prepare.py's shared values) and a NaN-gradient guard (skips, rather than corrupts, on a rare non-finite gradient). Decisive metric: validation_rmse_norm_mean.

## `test.py`

Evaluates the fold-checkpoint ensemble on both official test sequences:
```bash
python test.py
python test.py --checkpoint-set best_so_far
```

Reports RMSE in volts, separately for test_multisine and test_sinesweep, matching the benchmark's own convention. Monitoring only -- never used to choose between candidates.

## Typical Workflow

1. Read WienerHammersteinProcessNoise_description.md.
2. Place `WienerHammersteinFiles/` next to `prepare.py` (see Data setup above), then run `prepare.py` (do not modify).
3. Modify model.py and/or train.py's config_pars.
4. Run train.py, inspect plots/cross_validation_curves.png and the printed validation_rmse_norm_mean.
5. Optionally run test.py for monitoring only.
6. Repeat 3-5 for at least 15 total iterations, logging to results.tsv and search_journal.md.
7. Once a winner is chosen, run test.py --checkpoint-set best_so_far once for the final reported numbers (both test signals).

## Setting Up a New Experiment

Agree on a run tag, branch autoresearch/<tag>, read WienerHammersteinProcessNoise_description.md + prepare.py + model.py + train.py, initialize results.tsv (header only), reset search_journal.md to iteration 0.

## Experiment Rules

### What You Can Modify

- model.py
- train.py (config_pars and the training loop)

### What You Cannot Modify

- prepare.py -- read-only, no exceptions.
- No new dependencies.

### Target Metric

Lowest possible validation_rmse_norm_mean (5-fold CV, normalized units). The official per-signal test RMSEs (V, from test.py) are for reporting only.

## Search Strategy

Minimum 15 total iterations before concluding the search -- a hard floor, not a soft target. A run stopping early is premature regardless of how conclusive results look.

The fold design itself is no longer an open question for this search: the default 5-fold/10-realization design was already compared against alternatives (a 2-realization literature-standard set, a different fold grouping, and no held-out fold at all) and confirmed as the better choice based on the cross-validation curves and test-signal fits. This comparison is documented in search_journal.md and should not be repeated as one of the 15 iterations.

Fallback: if, over the course of the search, the 5-fold design starts showing clear symptoms of being unsuitable (e.g. validation_rmse_norm_mean persistently diverging from test.py's per-signal RMSEs in a way architecture changes can't explain), pause and re-open the fold-design question as an explicit, separately logged experiment before continuing -- rather than silently working around it.

Explore-then-refine for the rest of the search:
1. First 6-7 runs: meaningfully different model families (black-box RNN/GRU/LSTM, grey-box WIENERHAMMERSTEIN, other architectures if relevant).
2. Refine the most promising one through moderate hyperparameter changes.
3. If refinements plateau, try a different family, or the explicit process-noise-modeling idea from WienerHammersteinProcessNoise_description.md Section 5, before concluding.
4. Repeat until the 15-iteration minimum is met AND refinements have genuinely plateaued -- both conditions, not just the first one reached.

Notes specific to this benchmark:
- Since training data is noisy but test data is noiseless, a candidate with training loss much lower than validation/test performance may be overfitting to specific noise realizations rather than learning the underlying deterministic system -- worth checking directly.
- The additional measurement-campaign files cached under data/ (see WienerHammersteinProcessNoise_description.md Section 2) are available if a candidate wants to experiment with using more (noisy) training data -- a legitimate axis to explore, though it moves away from literature-comparable results, so document clearly if tried.
- n_taps_r/n_taps_s on WIENERHAMMERSTEIN: if a candidate's fit looks capacity-limited, try larger values before concluding the FIR approach itself is unsuitable.

## Logging Results

results.tsv, tab-separated: commit val_RMSE test_RMSE status description. val_RMSE = validation_rmse_norm_mean (decisive); test_RMSE = record as multisine=X;sinesweep=Y (V) from test.py, logged every run (monitoring only). results.tsv is a local log, untracked by Git.

## Experiment Loop

1. Check the current Git state.
2. First run of a new experiment: baseline, unmodified.
3. Every later run: modify train.py/model.py with one experimental idea; commit the change (non-baseline runs only).
4. Run train.py.
5. Inspect plots/cross_validation_curves.png and the printed validation_rmse_norm_mean.
6. Optionally run test.py for the monitoring-only per-signal RMSEs.
7. Decide keep/discard using validation_rmse_norm_mean only.
8. Update results.tsv and search_journal.md.
9. If discard, reset to the best kept commit before the next candidate.
10. Do not conclude before at least 15 total iterations.
11. Once satisfied with a winner, run test.py --checkpoint-set best_so_far once and record that as the final reported numbers.

## Search Journal Rules

Explanatory entries (what changed, why, what happened, kept or discarded, next step), not a terse ledger. Start fresh at iteration 0 for each new run tag.

## AI Agent Rules

- Read program.md AND WienerHammersteinProcessNoise_description.md before starting. Do not assume any model family already wins -- reason from the physical description and let validation_rmse_norm_mean decide.
- Treat prepare.py as strictly read-only.
- Use validation_rmse_norm_mean (5-fold CV), and only that metric, to decide what to keep. test.py (both test signals) is for monitoring only.
- Update results.tsv and search_journal.md after each run.
- Do not conclude the search before at least 15 total iterations (baseline included) -- stopping early is premature regardless of how good the current winner looks.
- Create Git commits for non-baseline experiments; reset to the best kept commit after every discard.
- Once satisfied, run test.py --checkpoint-set best_so_far once and record that single final result (both test signals).
- Continue autonomously unless a real blocker requires user input.