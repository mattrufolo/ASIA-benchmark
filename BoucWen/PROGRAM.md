# Bouc-Wen Hysteretic Benchmark Program

This folder contains the material used to run autoresearch on the
Bouc-Wen hysteretic benchmark. The goal is to train and evaluate models
that describe the displacement from the force input, for a system with
genuine hysteresis (history-dependent nonlinearity).

## IMPORTANT: Read This Before Starting

**Minimum 15-20 total iterations before concluding the search.** This is
a hard floor, not a soft target -- do not stop early just because a
result looks conclusive.

**No git.** This project does not use git branches, commits, or
checkouts as part of the search loop -- see "Keeping and Discarding
Experiments" below for how `keep`/`discard` decisions are tracked
instead, using plain file copies confined to this folder. Do not run any
`git` command as part of this search, and do not create a branch for it.

**Data setup (required before running `prepare.py`)**: this benchmark
provides NO official training data (confirmed from the paper itself --
see `BoucWen_description.md` Section 2); `prepare.py` self-generates it.
The two OFFICIAL test signals (multisine, sine-sweep) ARE real and
required: copy the extracted official zip's `BoucWenFiles/` folder next
to `prepare.py`, so `BoucWenFiles/Test signals/Validation signals/
{u,y}val_{multisine,sinesweep}.mat` exist. See `prepare.py`'s own
`RAW_MAT_DIR` comment for the exact expected layout.

```
step 1: SEARCH   train.py   -- leave-one-fold-out CV over 5 contiguous
                                folds of the self-generated training
                                recording (40960 samples)
step 2: TEST     test.py    -- evaluate the ENSEMBLE of the 5 fold checkpoints
                                on BOTH REAL official test signals (multisine,
                                sine-sweep), reported separately, simulation
                                mode always + prediction mode where supported
```

Start from first principles, not assumed conclusions. No model family
should be assumed to win or lose ahead of time. Reason from
`BoucWen_description.md`'s physical description, then let
`validation_rmse_norm_mean` decide.

## Benchmark Description

Read [BoucWen_description.md](./BoucWen_description.md) before starting:
it covers the hysteresis physics, why there's no official training data
and how this project generates its own, the real test data, the
numerical-stability fixes already applied, and the simulation-vs-
prediction mode distinction the benchmark itself asks for.

## Folder Structure

```
prepare.py
model.py
train.py
test.py
BoucWen_description.md
BoucWenFiles/            <- you place this (see Data setup above)
best_kept/                <- plain-file backup of the current best model.py/train.py (see below)
  model.py
  train.py
```

## `prepare.py`

Self-generates the estimation data (RK4 integration of the true Bouc-Wen
ODE, following the paper's own recipe -- see
`BoucWen_description.md` Section 3), and reads the two REAL official test
signals directly from the local `BoucWenFiles/` folder. Splits the
self-generated recording into 5 contiguous folds. Builds initial-condition
vectors (`history_window = 300`). **Should not be modified.** (Note: this
file takes ~30s to run due to the RK4 simulation step -- this is expected,
not a hang.)

## `model.py`

Contains a black-box recurrent baseline (`RNN`/`GRU`/`LSTM`, with an
optional `use_output_feedback=True` flag enabling prediction-mode
evaluation -- see `BoucWen_description.md` Section 5) and `BoucWenModel`
(`type: "BOUCWEN"`): the true Bouc-Wen physics, RK4-integrated with a
correctly-scaled `dt`, reference-scaled parameters, `nu` fixed at 1.
**Can be modified.**

**If adding a new grey-box model with its own discrete-time-critical
coefficient or integration step**, apply the same `dt = Ts/num_substeps`
convention from the start (Section 4 of the description) -- an earlier
version of this project used `dt = 1/num_substeps` and found severe
numerical instability as a direct result.

## `train.py`

Leave-one-fold-out CV, 5 parallel workers. `config_pars` holds model/
training hyperparameters (**can be modified**), including optional
per-experiment `fold_time_budget_seconds`/`eval_every` overrides (default
`None` -> use `prepare.py`'s shared values) and a NaN-gradient guard.
Decisive metric: `validation_rmse_norm_mean`, always computed in
**simulation mode** (matching what `keep`/`discard` decisions use).

## `test.py`

Evaluates the fold-checkpoint ensemble on both official test sequences:

```bash
python test.py
python test.py --checkpoint-set best_so_far
```

Reports RMSE in **meters**, separately for `test_multisine` and
`test_sinesweep`. Always reports **simulation mode**; if the checkpointed
model has `use_output_feedback=True`, ALSO reports **prediction mode**
(matching the benchmark's own explicit request). Monitoring only -- never
used to choose between candidates.

## Typical Workflow

1. Read `BoucWen_description.md`.
2. Place `BoucWenFiles/` next to `prepare.py` (see Data setup above),
   then run `prepare.py` (do not modify).
3. Modify `model.py` and/or `train.py`'s `config_pars`.
4. Run `train.py`, inspect `plots/cross_validation_curves.png` and the
   printed `validation_rmse_norm_mean`.
5. Optionally run `test.py` for monitoring only.
6. Repeat 3-5 for at least 15-20 total iterations, logging to
   `results.tsv` and `search_journal.md`.
7. Once a winner is chosen, run `test.py --checkpoint-set best_so_far`
   once for the final reported numbers (both test signals, both modes
   where applicable).

## Setting Up a New Search

No branch, no run tag. Read `BoucWen_description.md` + `prepare.py` +
`model.py` + `train.py`. Initialize `results.tsv` (header only) and
`search_journal.md` (empty) if they don't already exist -- if a previous
search already left these populated and you're deliberately starting a
fresh search rather than continuing the existing one, clear them first
and say so explicitly in the first `search_journal.md` entry. Create the
`best_kept/` folder (empty is fine) if it doesn't exist.

## Experiment Rules

### What You Can Modify

- `model.py`
- `train.py` (`config_pars` and the training loop)

### What You Cannot Modify

- `prepare.py` -- read-only, no exceptions.
- No new dependencies.
- The evaluation protocol (5-fold CV over the self-generated recording,
  `history_window=300`, the real official test signals) must not be
  changed.

### Target Metric

Lowest possible `validation_rmse_norm_mean` (5-fold CV, normalized
units, **simulation mode**). The official per-signal test RMSEs (from
`test.py`, both modes where applicable) are for reporting only.

## Search Strategy

**Minimum 15-20 total iterations before concluding the search -- a hard
floor, not a soft target.** Do not stop early just because a candidate's
result looks conclusive.

The search should prioritize changes that have a realistic chance of producing substantial improvements.

Do not spend too much time on minor parameter nudges such as:

- slightly changing `weight_decay`
- changing the width of a hidden layer by a very small amount
- other very small hyperparameter adjustments that are unlikely to move the metric meaningfully

These minor refinements are still allowed, but they should be used only occasionally, for example after a genuinely promising model family has already been identified and you want to refine it.

As a general strategy:

1. In the first 6–7 runs, explore meaningfully different model architectures and training objectives.
2. When a promising architecture is found, spend a few runs refining it through moderate hyperparameter or training-loop changes.
3. If these refinements do not produce meaningful validation-RMSE improvements, stop refining that architecture and try a different model family or substantially different modeling idea.
4. Repeat this explore/refine cycle: explore broad alternatives, then refine only the candidates that show clear promise.

Notes specific to this benchmark:

- Try `num_substeps` on `BOUCWEN` explicitly (1, 5, 10+): more substeps
  costs more compute per step but may improve integration accuracy --
  worth a direct comparison rather than assuming a value.
- `use_output_feedback=True` on the black-box models is worth testing
  explicitly, both for its own validation RMSE AND because it's the only
  way to get `test.py`'s prediction-mode numbers, which the benchmark
  itself asks for as a secondary figure of merit.
- If a `BOUCWEN` candidate's training loss freezes near its untrained
  value from epoch 0, suspect the same class of `dt`/integration-step
  bug documented in `BoucWen_description.md` Section 4 before concluding
  the architecture is unsuitable.
- Given the internal hysteretic state `z` is fundamentally unmeasurable
  (Section 6, challenge 2), a candidate that tries to infer/reconstruct
  it (rather than just fitting `y` end-to-end) is a genuinely different,
  worthwhile idea to test -- e.g. a latent-state model in the spirit of
  the EMPS project's `LATENT` model, replacing the hand-specified
  Bouc-Wen `z` dynamics with a free-form learned recursion while keeping
  the exact `y`/`ydot` mechanical equation.

## Keeping and Discarding Experiments

**Everything here is a plain file copy, confined to this project's own
folder. No git command is ever used.** This deliberately replaces an
earlier git-branch-based version of this workflow, after a real incident
where a `git checkout <branch> -- <path>` run from the wrong working
directory, combined with `checkpoints/` being gitignored, caused
checkpoints to be silently lost across MULTIPLE benchmark folders in a
shared repo at once. Plain `cp`, scoped to this one folder, cannot do
that.

- **`best_kept/model.py`, `best_kept/train.py`**: a plain-file snapshot
  of the current best-known configuration. Not a git commit -- just a
  copy sitting in this folder.
- **After a `keep` decision**: copy the CURRENT `model.py`/`train.py`
  (the ones that just produced this result) into `best_kept/`, e.g.:
  ```bash
  cp model.py best_kept/model.py
  cp train.py best_kept/train.py
  ```
- **After a `discard` decision**: copy `best_kept/model.py`/
  `best_kept/train.py` back over the working `model.py`/`train.py`
  before starting the next experiment, e.g.:
  ```bash
  cp best_kept/model.py model.py
  cp best_kept/train.py train.py
  ```
- **Checkpoints** (`checkpoints/`, `checkpoints/best_so_far/`) are
  already managed entirely by `train.py` itself via plain file copies
  (see `maybe_update_best_so_far` in `train.py`) -- nothing to do here,
  this was never the part of the workflow that broke.
- The **very first (baseline) run** has nothing to back up yet -- after
  its `keep` decision (baseline is always initially kept, being the only
  candidate so far), just create the first `best_kept/` snapshot as
  above.

## Logging Results

`results.tsv`, tab-separated: `commit  val_RMSE  test_RMSE  status  description`.
Since there is no git, the `commit` column is not a git hash -- use a
short sequential label instead (e.g. `run01`, `run02`, ...) matching
`search_journal.md`'s own iteration numbering.
`val_RMSE` = `validation_rmse_norm_mean` (decisive, simulation mode);
`test_RMSE` = record as `multisine_sim=X;sinesweep_sim=Y` (m), plus
`multisine_pred=..;sinesweep_pred=..` if the candidate supports
prediction mode, from `test.py` (monitoring only).

## Experiment Loop

1. First run: baseline, `model.py`/`train.py` unmodified. Always `keep`
   (only candidate so far) -- copy them into `best_kept/` (see above).
2. Every later run: modify `train.py`/`model.py` with one experimental
   idea.
3. Run `train.py`.
4. Inspect `plots/cross_validation_curves.png` and the printed
   `validation_rmse_norm_mean`.
5. Optionally run `test.py` for the monitoring-only per-signal RMSEs.
6. Decide `keep`/`discard` using `validation_rmse_norm_mean` only.
7. Update `results.tsv` and `search_journal.md`.
8. If `keep`: copy the current `model.py`/`train.py` into `best_kept/`.
   If `discard`: copy `best_kept/model.py`/`best_kept/train.py` back
   over the working files before the next experiment.
9. Do not conclude before at least 15-20 total iterations.
10. Once satisfied with a winner, run `test.py --checkpoint-set
    best_so_far` once and record that as the final reported numbers.

## Search Journal Rules

Explanatory entries (what changed, why, what happened, kept or discarded,
next step), not a terse ledger.

## AI Agent Rules

- Read `program.md` AND `BoucWen_description.md` before starting. Do not
  assume any model family already wins -- reason from the physical
  description and let `validation_rmse_norm_mean` decide.
- Treat `prepare.py` as strictly read-only.
- modify only `model.py` and `train.py`
- **Never run any `git` command as part of this search** -- no commits,
  no branches, no checkouts. Use the plain-file `best_kept/` mechanism
  above for every keep/discard decision instead.
- Use `validation_rmse_norm_mean` (5-fold CV, simulation mode), and only
  that metric, to decide what to keep. `test.py` (both test signals,
  both modes where applicable) is for monitoring only.
- Update `results.tsv` and `search_journal.md` after each run.
- After every `keep`, copy `model.py`/`train.py` into `best_kept/`.
  After every `discard`, copy `best_kept/model.py`/`best_kept/train.py`
  back over the working files before continuing.
- Once satisfied, run `test.py --checkpoint-set best_so_far` once and
  record that single final result (both test signals, both modes where
  applicable).
- Continue autonomously unless a real blocker requires user input.
