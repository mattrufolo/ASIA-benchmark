# Coupled Electric Drives (CED) Benchmark Program

This folder contains the material used to run autoresearch on the Coupled
Electric Drives (CED) benchmark. The goal of the project is to train and
evaluate models that describe the rectified pulley-speed dynamics from the
combined motor voltage input, matching the official
`nonlinear_benchmarks.CED()` train/test split.

## IMPORTANT: Read This Before Starting

**No git.** This project does not use git branches, commits, or
checkouts as part of the search loop -- see "Keeping and Discarding
Experiments" below for how `keep`/`discard` decisions are tracked
instead, using plain file copies confined to this folder. Do not run any
`git` command as part of this search, and do not create a branch for it.

This project previously used 4-fold leave-one-fold-out cross-validation.
That design was replaced because it systematically misled model selection:
every architecture that improved 4-fold CV RMSE made the official test
RMSE *worse*. The official test window is a calmer, qualitatively
different continuation of the recording than the busy middle of the train
data, and none of the old interior folds ever tested a model's ability to
extrapolate to that kind of segment. See `CED_description.md` Section 5
for the full analysis.

**The procedure now has three explicit, sequential steps. Do not skip or
reorder them, and do not blend their roles:**

```
step 1: SEARCH   train.py   -- train on [10:350], validate on [350:400] (both regimes)
step 2: REFIT    refit.py   -- retrain the SAME config on [10:400] (both regimes, no val)
step 3: TEST     test.py    -- evaluate the refit model ONCE on the official [400:500]
```

The single rule that must never be broken: the `[350:400]` window is used
**either** to choose between candidate models (step 1) **or** as training
data for the one final model (step 2) -- **never both**. Once a config has
been chosen using step 1 and refit in step 2, `[350:400]` must not be
re-examined to reconsider that choice.

## Benchmark Description

The benchmark is described in [CED_description.md](./CED_description.md).
Read it before starting: it explains why only 2 of the report's 5 raw
sequences are part of the official split, the Wiener/Wiener-Hammerstein
structure of the system, and the full rationale for the search/refit/test
procedure above.

## Folder Structure

The main files are:

- `prepare.py`
- `model.py`
- `train.py`
- `refit.py`
- `test.py`
- `CED_description.md`
- `best_kept/` -- plain-file backup of the current best `model.py`/`train.py` (see "Keeping and Discarding Experiments" below)

Below is the role of each file.

## `prepare.py`

The file `prepare.py` is responsible for:

- loading the official benchmark data (`low_amplitude` / `high_amplitude`
  realizations only)
- building 4 sets of cached sequences per realization:
  - `search_train_sequences`: `[history_window:350]`, used only by `train.py`
  - `search_val_sequences`: `[350:400]`, used only by `train.py`
  - `refit_train_sequences`: `[history_window:400]` (full data, no split),
    used only by `refit.py`
  - `test_sequences`: the official `[400:500]` split, used only by `test.py`
- building the initial-condition vectors from the previous 10 samples
  (`history_window = 10`, chosen to match the official test warmup)
- saving preprocessed data in cache

This file also contains general configuration parameters that **should not
be changed**, including `search_train_stop` (350) and `search_val_stop`
(400), which fix the search/val boundary described above.

In short, `prepare.py` prepares everything needed before training, for all
three steps at once.

## `model.py`

The file `model.py` contains the model architectures used to describe the
benchmark dynamics: a black-box recurrent baseline (`RNN`/`GRU`/`LSTM`), a
grey-box `WIENER` model (a learnable companion-form linear ODE followed by a
smooth output rectifier, matching eqs. 6-8 of the technical report), and a
`HYBRID` model combining the two.

This file **can be modified**. It is the correct place to work if you want
to change or improve the model architecture — different recurrent
families, different ODE orders, different output nonlinearities, residual
or hybrid variants, and so on.

## `train.py` (step 1: SEARCH)

The file `train.py` runs the SEARCH-phase training: it trains one model on
`search_train_sequences` (both regimes jointly, `[history_window:350]`) and
validates it on `search_val_sequences` (both regimes jointly, `[350:400]`).

At the beginning of the file there is a dictionary called `config_pars`,
which contains the main training hyperparameters, for example:

- learning rate
- recurrent network type / model type
- number of epochs
- number of hidden states
- hidden layer sizes
- activation function
- number of layers
- dropout
- direct feedthrough

These parameters **can be modified**. They are one possible experimentation
space, but they are not privileged over `model.py`: the agent is free to
change the model architecture, the training setup, or both, depending on
what seems most promising.

The reported metric is a single scalar `validation_rmse_norm`, computed by
concatenating both regimes' held-out `[350:400]` windows -- there is no
fold averaging anymore. This is the **only** metric that should drive
architecture/hyperparameter decisions.

Read the current version of `train.py` to better understand how the file
is organized and which metrics are used.

## `refit.py` (step 2: REFIT)

The file `refit.py` retrains **whatever configuration currently sits in
`train.py`'s `config_pars`** on the full `[history_window:400]` of both
regimes -- no held-out split. It should be run exactly once per experiment,
after (not during) the search phase, once a winning config has been
identified using `train.py`'s search-validation RMSE.

Because there is no held-out data left at this stage, `refit.py` cannot
early-stop on a validation metric. Instead, by default, it watches its own
full-window **training** RMSE (evaluated on the fixed `[history_window:400]`
sequences, not the noisy random training window) and stops once that
plateaus (patience-based), bounded by `refit_time_budget_seconds` as a
safety net. This is preferred over deriving a fixed epoch count from the
search run's `best_epoch`: since both phases now train on randomly-sampled
windows (see `prepare.sample_training_window`), search's `best_epoch` is
itself sensitive to which random windows it happened to draw, and refit
samples from a wider range (`[10:400]` vs. search's `[10:350]`) -- so a
literal epoch-count transplant across the two is not a reliable proxy for
"trained until converged." An explicit `--epochs` override is still
available for exact reproducibility or quick manual experiments, bypassing
the plateau logic entirely.

`refit.py` should be treated the same way as `prepare.py` during
autoresearch: it is not part of the thing being tuned, and its own logic
should not need to change between experiments. What changes between
experiments is only which config it happens to retrain, via `train.py`.
Because it is never itself modified, it has no `best_kept/` counterpart --
see "Keeping and Discarding Experiments" below.

## `test.py` (step 3: TEST)

The file `test.py` evaluates a single checkpoint on the two official test
realizations (`test_low_amplitude`, `test_high_amplitude`). By default it
evaluates `checkpoints/final/model.pt` (produced by `refit.py`); if that
does not exist yet, it falls back to the SEARCH checkpoint with an explicit
warning, purely so the val/test gap can still be monitored during the
search phase.

`test.py` **cannot be used to make search decisions**. It reports RMSE per
realization in the official `[test_low; test_high]` order and units
(ticks/s), matching `nonlinear_benchmarks/submission_examples/CED.py`.

## Training Log

During SEARCH training, `train.py` writes one log file,
`logs/train_search.log`, showing the evolution of the training loss and of
the training/validation RMSE (both regimes concatenated, not per-fold).

The time budget for the search run is fixed at
**search_time_budget_seconds seconds**, specified in `prepare.py`. If the
budget is reached, training should stop, keep the best validation
checkpoint reached, and finish. Hitting the budget is not a failure; it is
a normal stopping condition.

## Typical Workflow

To use this folder correctly, the recommended workflow is:

1. Read `CED_description.md` to understand the benchmark and the
   search/refit/test procedure.
2. Run `prepare.py` to load and preprocess the datasets. This file should
   not be modified.
3. Modify `model.py` and/or `train.py`'s `config_pars` depending on the
   experimental idea you want to test.
4. Run `train.py` (step 1: SEARCH) and inspect `logs/train_search.log` and
   the printed `validation_rmse_norm`.
5. Repeat steps 3-4 for as many candidates as the search strategy calls for
   (see below), recording each in `results.tsv` and `search_journal.md`.
6. Once a winning config has been identified purely from step-1
   `validation_rmse_norm` values, run `refit.py` **once** for that config.
7. Run `test.py` to get the official (reporting-only) test RMSE for that
   refit model.

## Setting Up a New Experiment

To set up a new experiment, work with the user through the following
steps.

### 1. Read the In-Scope Files

Before starting, read the main files involved in the experiment
completely:

- `CED_description.md`
- `prepare.py` (cannot be modified)
- `model.py` (can be modified)
- `train.py` (can be modified)
- `refit.py` (cannot be modified)

### 2. Initialize `results.tsv` and `best_kept/`

At the beginning, create a file called `results.tsv` containing only the
header row.

At the same time, `search_journal.md` must also be reset for the new
search: it should either not exist yet or contain no iteration entries
from a previous search. Its iteration count must start again from `0` --
if a previous search already left `results.tsv`/`search_journal.md`
populated (including one that used the retired 4-fold procedure) and this
is deliberately a fresh search rather than a continuation, clear them
first and say so explicitly in the first `search_journal.md` entry.

Create the `best_kept/` folder (empty is fine) if it doesn't already
exist.

### 3. Confirm the Setup

Before launching the first run, confirm that:

- the relevant files have been read
- `results.tsv` has been initialized
- `search_journal.md` has been reset for the new experiment and will start again from iteration `0`
- `best_kept/` exists


Once the setup is confirmed, experimentation can begin.

## Experiment Rules

The training script is launched as:

```bash
python train.py
```

In this project, the main objective is to improve the **step-1 SEARCH
validation RMSE** (`validation_rmse_norm`, both regimes concatenated over
`[350:400]`). This is the reference metric to optimize during the
explore/refine loop below. `refit.py` and `test.py` are run only after a
winning config has already been chosen this way -- not as part of choosing
between candidates.

### What You Can Modify

Only the following files should be edited during experiments:

- `model.py`
- `train.py` (specifically its `config_pars` and training loop)

In these files, you are free to modify:

- model architecture
- model type
- optimizer
- training hyperparameters
- training loop
- model size
- any other training-related choice implemented inside these two files

### What You Cannot Modify

- `prepare.py` must not be modified. It should be treated as read-only.
- `refit.py` must not be modified. It always retrains whatever `train.py`
  currently specifies; it should not itself be tuned.
- no new packages or dependencies should be installed
- only the packages already available in the current project may be used
- the evaluation protocol (the `[10:350]`/`[350:400]`/`[10:400]`/`[400:500]`
  split defined in `prepare.py`) must not be changed
- the training data must remain restricted to the 2 official realizations
  loaded by `nonlinear_benchmarks.CED()`; do not substitute the PRBS-only
  sequences from the technical report, since they are not part of the
  official leaderboard split

### Target Metric

The goal is simple:

- obtain the lowest possible **step-1 SEARCH validation RMSE**
  (`validation_rmse_norm`)

This is the metric that should guide hyperparameter selection and model
changes. It is computed once per candidate, on the single held-out
`[350:400]` window (both regimes concatenated) -- there is no fold
averaging to reconcile.

The official test RMSE (from `test.py`, evaluated on a checkpoint that
`refit.py` produced) must not be used to decide which model to keep during
the search loop. It is only computed once, at the very end, for reporting.

### Runtime and Practical Constraints

The training code must:

- run without crashing
- complete correctly on the available hardware
- remain compatible with the current project setup

## Search Strategy

The search should prioritize changes that have a realistic chance of
producing substantial improvements.

Do not spend too much time on minor parameter nudges such as:

- slightly changing `weight_decay`
- changing the width of a hidden layer by a very small amount
- other very small hyperparameter adjustments that are unlikely to move
  the metric meaningfully

These minor refinements are still allowed, but they should be used only
occasionally, for example after a genuinely promising model family has
already been identified and you want to refine it.

As a general strategy:

1. In the first 6-7 runs, explore meaningfully different model
   architectures and training objectives.
2. When a promising architecture is found, spend a few runs refining it
   through moderate hyperparameter or training-loop changes.
3. If these refinements do not produce meaningful validation-RMSE
   improvements, stop refining that architecture and try a different model
   family or substantially different modeling idea.
4. Repeat this explore/refine cycle.
5. Only once the explore/refine loop has converged on a winner by
   `validation_rmse_norm`, run `refit.py` once and `test.py` once for the
   final reported number.

The agent should therefore be willing to test:

- different neural architectures (RNN/GRU/LSTM, TCN, attention, ...)
- different orders for the grey-box `WIENER` ODE, and different pole
  initializations
- residual or direct-feedthrough variants
- hybrid physical + learned-residual models
- different treatment of the output rectifier (hard `|.|` vs. smooth
  surrogates such as `sqrt(x^2 + eps)`)
- other materially different modeling choices

A note specific to this benchmark: grey-box/physical model types
(`WIENER`, `HYBRID`) integrate an ODE step-by-step in normalized signal
space. A bad pole/step-size initialization can make the very first
forward pass diverge to `inf`/`NaN` before any training happens. If a
candidate produces `NaN` losses from epoch 0, first check the
initialization of the state-transition parameters (companion-form
coefficients, `dt`) before concluding the architecture itself is
unsuitable.

A second note, specific to this benchmark's search/val split: because
`search_val_sequences` are only 50 samples per regime (100 total), the
step-1 metric is noisier than the old 4-fold aggregate was. Do not
over-interpret very small `validation_rmse_norm` differences between two
candidates as meaningful; prefer changes that produce a clear, sizeable
improvement.

### First Run

The first run of a new experiment should always be the baseline run,
executed with `train.py` in its current, unmodified form, so that future
changes can be compared against a clear reference result.

## Keeping and Discarding Experiments (no git)

**Everything here is a plain file copy, confined to this project's own
folder. No git command is ever used.** This deliberately replaces an
earlier git-branch-based version of this workflow, after a real incident
where a `git checkout <branch> -- <path>` run from the wrong working
directory, combined with `checkpoints/` being gitignored, caused
checkpoints to be silently lost across MULTIPLE benchmark folders in a
shared repo at once. Plain `cp`, scoped to this one folder, cannot do
that.

- **`best_kept/model.py`, `best_kept/train.py`**: a plain-file snapshot of
  the current best-known SEARCH-phase configuration. Not a git commit --
  just a copy sitting in this folder. (`prepare.py` and `refit.py` are
  never modified, so they have no `best_kept/` counterpart.)
- **After a `keep` decision**: copy the CURRENT `model.py`/`train.py`
  (the ones that just produced this SEARCH result) into `best_kept/`,
  e.g.:
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
- **Checkpoints** (`checkpoints/`, including `checkpoints/final/` produced
  by `refit.py`) are already managed entirely by the training scripts
  themselves via plain file copies -- nothing to do here, this was never
  the part of the workflow that broke.
- The **very first (baseline) run** has nothing to back up yet -- after
  its `keep` decision (baseline is always initially kept, being the only
  candidate so far), just create the first `best_kept/` snapshot as above.
- Once the explore/refine loop concludes and `refit.py`/`test.py` have
  been run for the final chosen config, `best_kept/` should already hold
  exactly that config (it was the last thing kept) -- nothing further to
  copy at that point.

## Logging Results

When an experiment is completed, record it in `results.tsv`.

This file should be **tab-separated**, not comma-separated, with columns:

```text
commit	val_RMSE	test_RMSE	status	description
```

The columns mean:

1. `commit`: since there is no git, this is not a Git hash -- use a short
   sequential label instead (e.g. `run01`, `run02`, ...), matching
   `search_journal.md`'s own iteration numbering
2. `val_RMSE`: the SEARCH-phase `validation_rmse_norm` from `train.py`
   (normalized units, `[350:400]` of both regimes concatenated) -- this is
   the decisive metric
3. `test_RMSE`: for most rows, this should be left as `NA`, since
   `test.py` should not normally be run during the search loop (it can only
   evaluate a SEARCH-stage checkpoint, with the explicit fallback warning,
   for occasional monitoring of the val/test gap -- never as a
   keep/discard signal). Only the row for the final chosen config, after
   `refit.py` + `test.py` have been run, should carry a real `test_RMSE`
   value (the denormalized `submission_rmse_mean`, ticks/s), and that row
   should be clearly marked in `description` as the final refit result.
4. `status`: typically `keep` or `discard`; use `final` for the one row
   that corresponds to the post-refit, reported result
5. `description`: a short explanation of what the experiment changed

If a run crashes and no valid result is produced, you may record `inf` as
the RMSE value, `discard` as the status, and a short crash description.

Example:

```text
commit	val_RMSE	test_RMSE	status	description
run01	0.612400	NA	keep	baseline LSTM n_hidden=16 (search phase only)
run02	0.910200	NA	discard	WIENER diverges to NaN, pole init too aggressive
run03	0.545100	NA	keep	GRU n_hidden=64 num_layers=2, new best search RMSE
run03	0.545100	0.198300	final	refit.py + test.py run on the GRU config above
```

`results.tsv` is meant to be a local experiment log, kept as a plain file
in this folder.

## Experiment Loop

The typical loop is:

1. If this is the first run of a brand-new search, run the baseline
   exactly as-is, without modifying `model.py` or `train.py`.
2. For every later run, modify `train.py` and/or `model.py` with one
   experimental idea.
3. Run the experiment using `python train.py` (step 1: SEARCH only).
4. Inspect `logs/train_search.log`.
5. Read the final `validation_rmse_norm` printed by `train.py`.
6. Decide whether the experiment should be marked as `keep` or `discard`,
   using `validation_rmse_norm` only.
7. Update `results.tsv` with `val_RMSE` set to `validation_rmse_norm` and
   `test_RMSE` left as `NA` for this row.
8. Update `search_journal.md` with an explicit note describing what was
   changed, why, what happened, and what should be tried next.
9. If `keep`: copy the current `model.py`/`train.py` into `best_kept/`.
   If `discard`: copy `best_kept/model.py`/`best_kept/train.py` back over
   the working files before starting the next experiment.
10. Continue with the next experiment unless a real blocker requires a
    decision from the user.
11. Once the explore/refine loop concludes and a final config is chosen,
    run `refit.py` once and `test.py` once, and add one final row to
    `results.tsv` (status `final`) with the real `test_RMSE`.

## Search Journal Rules

`search_journal.md` should be explanatory, not just a terse ledger. After
every completed run, write short but explicit sentences that state:

- what was changed in `model.py` and/or `train.py`
- why that change was chosen, meaning the hypothesis behind it
- what happened in the run, including the important metric or training
  behavior
- whether the run was kept or discarded
- what the next step should be

The journal should make it easy for a reader to understand the reasoning
of the search loop without having to infer it from a code diff alone. If
an older journal used the retired 4-fold CV procedure, do not append to it
directly: start a fresh journal (iteration 0), and note once, at the top,
that earlier entries used a different (now retired) validation design and
are not numerically comparable.

## AI Agent Rules

If an AI agent is launched to run the experiment loop, it should follow
these rules:

- read `program.md` before starting
- treat `prepare.py` and `refit.py` as read-only
- modify only `model.py` and `train.py`
- **never run any `git` command as part of this search** -- no commits,
  no branches, no checkouts. Use the plain-file `best_kept/` mechanism
  above for every keep/discard decision instead
- use `train.py`'s `validation_rmse_norm` (step 1: SEARCH), and only that
  metric, to decide what to keep during the explore/refine loop
- never run `refit.py`/`test.py` as part of choosing between candidates;
  only after a winning config has already been chosen
- use `logs/train_search.log` to inspect training behavior
- update `results.tsv` after each completed SEARCH run (`test_RMSE = NA`
  unless the row is the final post-refit result)
- update `search_journal.md` after each completed run
- after every `keep`, copy `model.py`/`train.py` into `best_kept/`
- after every `discard`, copy `best_kept/model.py`/`best_kept/train.py`
  back over the working files before continuing
- once satisfied with a winner, run `refit.py` once, then `test.py` once,
  and record that single final result clearly in both `results.tsv` and
  `search_journal.md`
- continue autonomously unless a real blocker requires user input