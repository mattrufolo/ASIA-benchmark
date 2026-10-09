# EMPS Benchmark Program

This folder contains the material used to run autoresearch on the EMPS
(Electro-Mechanical Positioning System) benchmark. The goal of the project is to train
and evaluate models that describe the system dynamics (motor force `vir` -> load
position `qm`) from input-output data.

## IMPORTANT: Read This Before Starting

**No git.** This project does not use git branches, commits, or checkouts as part of
the search loop -- see "Keeping and Discarding Experiments" below for how
`keep`/`discard` decisions are tracked instead, using plain file copies confined to
this folder. Do not run any `git` command as part of this search, and do not create a
branch for it.

## Benchmark Description

The benchmark is described in [EMPS_description.md](./EMPS_description.md), which
summarizes the reference paper (Janot, Gautier, Brunot, 2019).

## Folder Structure

The main files are:

- `prepare.py`
- `model.py`
- `train.py`
- `test.py`
- `EMPS_description.md`
- `best_kept/` -- plain-file backup of the current best `model.py`/`train.py` (see "Keeping and Discarding Experiments" below)

Below is the role of each file.

## `prepare.py`

The file `prepare.py` is responsible for:

- loading the official benchmark data (`nonlinear_benchmarks.EMPS()`),
- splitting the full official training trajectory (`DATA_EMPS.mat`) into exactly
  **one** contiguous "train" part (the first `train_fraction`, default 80%) and
  **one** "validation" part -- **no folds, no k-fold cross-validation**,
- building the initial-condition vectors from the previous 5 samples,
- saving preprocessed data in cache.

Important detail of the train/validation protocol: the "validation" sequence covers
the **entire** training trajectory, starting from the same point as "train". Its
`warmup` field is set to the length of the "train" part, so that:

- the model is simulated (run in its own closed loop) across the *whole* trajectory,
  including the part used for training,
- but the validation RMSE is computed using **only the last 20%** of that simulation
  (i.e. the part actually held out from the training loss).

This is intentional: it makes the validation score reflect how the model behaves when
it must run itself forward from its own state (as it also must on the official test
trajectory), rather than scoring it on a segment that was reset from ground truth.

This file also contains general configuration parameters that **should not be
changed**. In particular, it defines how the data are read, how the single
train/validation split is built, how initial conditions are constructed, and which
cached files are produced.

In short, `prepare.py` prepares everything needed before training.

## `model.py`

The file `model.py` contains the model architectures used to describe the benchmark
dynamics. It **can be modified**. Three model families are provided as a starting
point, corresponding to genuinely different modeling philosophies:

- **Black-box** (`type in {"RNN", "GRU", "LSTM", "LTC"}`): a generic recurrent model
  (or a closed-form continuous-time cell, `LTC`) with no knowledge of the EMPS physics.
- **White-box** (`type in {"PHYSICAL", "PHYSICAL_ASYM"}`): directly implements the
  paper's Direct Dynamic Model, `qdd = (vir - Fv*qd - Fc*sign(qd) - offset)/M`
  (`PHYSICAL`), or its asymmetric-friction variant from eq. (12)
  (`PHYSICAL_ASYM`, separate `Fv+/Fc+` and `Fv-/Fc-` for positive/negative velocity).
  Both support Euler or RK4 integration (`integrator` in the model config) using the
  true, known EMPS sampling time (not learned).
- **Grey-box / hybrid** (`type = "HYBRID"`): the white-box physical backbone above
  (optionally with asymmetric friction, `asymmetric_friction: True`) plus a small
  LSTM residual correction for unmodeled effects (nested-PD tracking error, encoder
  quantization, mechanical resonances, ...).

You are free to add further variants inside `model.py` (e.g. a second-order
mechanical-resonance term, a learned friction curve replacing `Fc*sign(qd)`, etc.).

## `train.py`

The file `train.py` runs a single training run (train on the first `train_fraction`,
evaluate on the held-out tail as described above).

At the beginning of the file there is a dictionary called `config_pars`, which
contains the main training hyperparameters, for example:

- `lr` (learning rate)
- `type` (which model family from `model.py` to use)
- `max_epochs`
- `n_hidden_states`, `hidden_sizes`, `activation`, `num_layers`, `dropout_prob`
- `direct_feedthrough`
- `integrator`, `asymmetric_friction` (physical/hybrid families only)
- `drift_penalty_weight`, `drift_penalty_window` (see below)

These parameters **can be modified**. They are one possible experimentation space, but
they are not privileged over `model.py`: the agent is free to change the model
architecture, the training setup, or both, depending on what seems most promising.

### Fixed-iteration training (no early stopping)

Unlike a typical CV setup, this benchmark's `train.py` **does not implement early
stopping**. Training always runs for exactly `max_epochs` iterations (the only
exception is `time_budget_seconds`, a wall-clock **safety net**, not a metric-based
stopping rule -- see below). This is intentional and should be preserved by every
experiment:

- Keep `lr` **small**. This is a stiff, friction-driven, closed-loop system; large
  steps tend to destabilize the physical/hybrid models in particular, and a small
  step size with a long fixed run lets the optimizer pick up slow/small patterns in
  the ~20s trajectories that an early-stopped run would likely miss.
- Keep `max_epochs` **large** so that a small-lr run can actually converge.
- Do not reintroduce a patience-based stop; if you want a learning-rate schedule, a
  simple non-metric-based schedule (e.g. a fixed step or cosine decay over the known
  `max_epochs`) is fine, but it must not depend on validation performance in a way
  that can cut the run short.

The training loop must still respect the fixed benchmark settings coming from
`prepare.py`, including `eval_every` and `time_budget_seconds`.

## `test.py`

The file `test.py` evaluates the saved checkpoint on the official test trajectory
(`DATA_EMPS_PULSES.mat`, exposed as `test` by `nonlinear_benchmarks.EMPS()`). It
cannot be used by autoresearch. It is only to assess final performance on test data,
and reports RMSE in millimeters (matching the official benchmark convention) as well
as in meters.

## Training Log

During training, the code writes a single log file, `logs/train.log`, showing the
evolution of the training loss (including the drift penalty term, if enabled), the
plain training MSE, and the training/validation RMSE (both in normalized units) at
every `eval_every` iterations.

## Typical Workflow

To use this folder correctly, the recommended workflow is:

1. Read `EMPS_description.md` to understand the benchmark.
2. Run `prepare.py` to load and preprocess the datasets. This file should not be
   modified.
3. Modify `model.py` and/or `train.py` depending on the experimental idea you want to
   test.
4. In `train.py`, update `config_pars` or other training logic if that is the most
   useful change for the current candidate.
5. Run `train.py` to train the model and inspect `logs/train.log`.
6. Run `test.py` only after the search decision has been made.

## Setting Up a New Search

To set up a new search, work with the user through the following steps.

### 1. Read the In-Scope Files

Before starting, read the main files involved in the experiment.

The repository is small, so the important files should be reviewed completely:

- `EMPS_description.md`
- `prepare.py` (cannot be modified)
- `model.py` (can be modified)
- `train.py` (can be modified)

### 2. Initialize `results.tsv` and `best_kept/`

At the beginning, create a file called `results.tsv` containing only the header row.

At the same time, `search_journal.md` must also be reset for the new search: it
should either not exist yet or contain no iteration entries from a previous search.
Its iteration count must start again from `0` -- if a previous search already left
`results.tsv`/`search_journal.md` populated and this is deliberately a fresh search
rather than a continuation, clear them first and say so explicitly in the first
`search_journal.md` entry.

Create the `best_kept/` folder (empty is fine) if it doesn't already exist.

### 3. Confirm the Setup

Before launching the first run, confirm that:

- the relevant files have been read
- `results.tsv` has been initialized
- `search_journal.md` has been reset and will start again from iteration `0`
- `best_kept/` exists

Once the setup is confirmed, experimentation can begin.

## Experiment Rules

The training script is launched as:

```bash
python train.py
```

In this project, the main objective is to improve the **validation RMSE** (the RMSE
computed on the last 20% of the training trajectory, as described above). This is a
single number per run -- there is no fold-aggregation step.

### Minimum Number of Iterations

Run **at least 15 iterations** (i.e. at least 15 completed, logged experiments,
including the baseline) before concluding the search. This benchmark's search space
(three qualitatively different model families, several friction/integrator variants,
and hybrid combinations) is large enough that fewer than 15 iterations is very
unlikely to have explored it adequately.

### What You Can Modify

Only the following files should be edited during experiments:

- `model.py`
- `train.py`

In these files, you are free to modify:

- model architecture / model type
- optimizer
- training hyperparameters (keeping `lr` small and `max_epochs` large, per above)
- the drift penalty settings
- model size
- any other training-related choice implemented inside these two files

### What You Cannot Modify

The following constraints should be respected:

- `prepare.py` must not be modified. It should be treated as read-only.
- no new packages or dependencies should be installed
- only the packages already available in the current project may be used
- the evaluation protocol must not be changed (single 80/20 split, no folds, no
  early stopping)

### Target Metric

The goal is simple:

- obtain the lowest possible **validation RMSE** (normalized units, as printed by
  `train.py` and stored in `checkpoints/training_summary.json`).

This is the main metric that should guide hyperparameter selection and model changes.
The test trajectory must not be used to decide which model to keep.

### Runtime and Practical Constraints

The training code must:

- run without crashing
- complete correctly on the available hardware
- remain compatible with the current project setup

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
## Keeping and Discarding Experiments 

**Everything here is a plain file copy, confined to this project's own folder. No git
command is ever used.** This deliberately replaces an earlier git-branch-based version
of this workflow, after a real incident where a `git checkout <branch> -- <path>` run
from the wrong working directory, combined with `checkpoints/` being gitignored,
caused checkpoints to be silently lost across MULTIPLE benchmark folders in a shared
repo at once. Plain `cp`, scoped to this one folder, cannot do that.

- **`best_kept/model.py`, `best_kept/train.py`**: a plain-file snapshot of the current
  best-known configuration. Not a git commit -- just a copy sitting in this folder.
- **After a `keep` decision**: copy the CURRENT `model.py`/`train.py` (the ones that
  just produced this result) into `best_kept/`, e.g.:
  ```bash
  cp model.py best_kept/model.py
  cp train.py best_kept/train.py
  ```
- **After a `discard` decision**: copy `best_kept/model.py`/`best_kept/train.py` back
  over the working `model.py`/`train.py` before starting the next experiment, e.g.:
  ```bash
  cp best_kept/model.py model.py
  cp best_kept/train.py train.py
  ```
- **Checkpoints** (`checkpoints/`, `checkpoints/best_so_far/`) are
 are already managed entirely by `train.py` itself
  via plain file copies -- nothing to do here, this was never the part of the
  workflow that broke.
- The **very first (baseline) run** has nothing to back up yet -- after its `keep`
  decision (baseline is always initially kept, being the only candidate so far), just
  create the first `best_kept/` snapshot as above.
- The search should always have a clear notion of the current best-kept
  configuration -- new experiments should build logically from what's in
  `best_kept/`, not from the latest discarded attempt.

## Logging Results

When an experiment is completed, record it in `results.tsv`.

This file should be **tab-separated**, not comma-separated.

The file should contain a header row and the following columns:

```text
commit	val_RMSE	test_RMSE	status	description
```

The columns mean:

1. `commit`: since there is no git, this is not a Git hash -- use a short sequential
   label instead (e.g. `run01`, `run02`, ...), matching `search_journal.md`'s own
   iteration numbering
2. `val_RMSE`: the validation RMSE (normalized units) from `train.py` /
   `checkpoints/training_summary.json`
3. `test_RMSE`: the test RMSE in mm reported by `test.py` on the checkpoint produced
   by the just-finished run; this is for monitoring only
4. `status`: typically `keep` or `discard`
5. `description`: a short explanation of what the experiment changed

If a run crashes and no valid result is produced, you may record:

- `inf` as the RMSE value
- `discard` as the status
- a short crash description in the text field

Example:

```text
commit	val_RMSE	test_RMSE	status	description
run01	0.997900	4.102300	keep	baseline (LSTM, small lr)
run02	0.812300	3.594800	keep	switch to PHYSICAL (symmetric friction, euler)
run03	0.780500	3.410500	keep	switch to PHYSICAL_ASYM (asymmetric friction)
run04	0.781200	3.415000	discard	PHYSICAL_ASYM + rk4 integrator, no improvement
run05	0.742100	3.220100	keep	HYBRID (PHYSICAL_ASYM backbone + LSTM residual)
run06	inf	NA	discard	crash due to shape mismatch
```

`results.tsv` is meant to be a local experiment log, kept as a plain file in this
folder.

## Experiment Loop

1. First run: baseline, `model.py`/`train.py` unmodified. Always `keep`
   (only candidate so far) -- copy them into `best_kept/` (see above).
2. Every later run: modify `train.py`/`model.py` with one experimental
   idea.
3. Run `train.py`.
4. Inspect `plots/training_curves.png` and the printed
   `validation RMSE`.
5. Optionally run `test.py` for the monitoring-only per-signal RMSEs.
6. Decide `keep`/`discard` using `validation RMSE` only.
7. Update `results.tsv` and `search_journal.md`.
8. If `keep`: copy the current `model.py`/`train.py` into `best_kept/`.
   If `discard`: copy `best_kept/model.py`/`best_kept/train.py` back
   over the working files before the next experiment.
9. Do not conclude before at least 15-20 total iterations.
10. Once satisfied with a winner, run `test.py --checkpoint-set
    best_so_far` once and record that as the final reported numbers.
## Search Journal Rules

`search_journal.md` should be explanatory, not just a terse ledger. After every
completed run, write short but explicit sentences that state:

- what was changed in `model.py` and/or `train.py`
- why that change was chosen, meaning the hypothesis behind it
- what happened in the run, including the important metric or training behavior
- whether the run was kept or discarded
- what the next step should be

The journal should make it easy for a reader to understand the reasoning of the
search loop without having to infer it from a code diff alone.

## AI Agent Rules

If an AI agent is launched to run the experiment loop, it should follow these rules:

- read `program.md` before starting
- treat `prepare.py` as read-only
- modify only `model.py` and `train.py`
- **never run any `git` command as part of this search** -- no commits, no branches,
  no checkouts. Use the plain-file `best_kept/` mechanism above for every
  keep/discard decision instead
- keep `train.py` as a single-run trainer
- use `logs/train.log` to inspect training behavior
- use validation RMSE, not test RMSE, to decide what to keep
- run at least 15 iterations before concluding the search
- follow the Phase 1 (explore families) / Phase 2 (tune the best family) shape
- update `results.tsv` after each completed run
- update `search_journal.md` after each completed run
- after every `keep`, copy `model.py`/`train.py` into `best_kept/`
- after every `discard`, copy `best_kept/model.py`/`best_kept/train.py` back over the
  working files before continuing
- continue autonomously unless a real blocker requires user input