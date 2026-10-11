# Parallel Wiener-Hammerstein (ParallelWH) Benchmark Program

This folder contains the material used to run autoresearch on the
Parallel Wiener-Hammerstein benchmark (Schoukens, Marconato, Pintelon,
Vandersteen & Rolain, *Parametric identification of parallel
Wiener-Hammerstein systems*, Automatica, vol. 51, pp.111-122, 2015). The
goal is to train and evaluate models that describe the output voltage
from the input voltage for a real electronic circuit made of TWO
Wiener-Hammerstein branches connected in parallel and summed.

## IMPORTANT: Read This Before Starting

**No git.** This project does not use git branches, commits, or
checkouts as part of the search loop -- see "Keeping and Discarding
Experiments" below for how `keep`/`discard` decisions are tracked
instead, using plain file copies confined to this folder. Do not run any
`git` command as part of this search, and do not create a branch for it.

**Structure**: each of the two parallel branches is a genuine
Wiener-Hammerstein system in its own right -- a diode-resistor static
nonlinearity sandwiched between two 3rd-order LTI filters -- and the two
branches' outputs are summed. The overall SISO system has 12th-order
linear dynamics (`n_x = 12`). The two LTI blocks per branch, PLUS the
parallel-branch structure itself, is what makes this harder to identify
than the plain (single-branch) `WienerHammerBenchMark` project in this
collection -- see `ParallelWH_description.md` for the full physical
description.

**Data and folds**: 100 training sequences (20 excitation phases x 5
amplitudes, ~100mV-1V) and 5 official test sequences (one held-out phase
per amplitude), `fs ~= 78kHz`. Folds are grouped by PHASE, not by
amplitude -- **do not change this**:

```
step 1: SEARCH   train.py   -- leave-one-fold-out CV over 5 folds,
                                each fold = 4 phases x 5 amplitudes
                                (20 sequences), grouped by PHASE GROUP
step 2: TEST     test.py    -- evaluate the ENSEMBLE of the 5 fold checkpoints
                                on the 5 official held-out-phase test sequences
```

**Why phase-grouped, not amplitude-grouped folds**: the official test
task is specifically "generalize to an unseen excitation phase at a
familiar amplitude" -- amplitude-based folds would validate the wrong
axis of generalization (they'd test amplitude extrapolation, which isn't
what the official test set actually asks for). This was a deliberate
design decision made when this project was first built and should not be
revisited without a clear reason logged in `search_journal.md`.

**Worth double-checking against your actual data**: some published
descriptions of this benchmark's official zip also mention a third,
increasing-amplitude "arrow" test signal (linearly growing-amplitude
Gaussian noise), similar in spirit to the Silverbox project's own arrow
test in this collection. If your `prepare.py`/data folder includes this
signal, it should be evaluated (and reported) the same way the 5
phase-held-out sequences are -- separately, monitoring only, never used
for `keep`/`discard` decisions.

Start from first principles, not assumed conclusions. No model family
should be assumed to win or lose ahead of time. Reason from
`ParallelWH_description.md`'s physical description, then let
`validation_rmse_norm_mean` decide.

## Benchmark Description

Read [ParallelWH_description.md](./ParallelWH_description.md) before
starting: it covers the two-branch physical structure, the phase/
amplitude data design, and the fold-grouping rationale.

## Folder Structure

```
prepare.py
model.py
train.py
test.py
ParallelWH_description.md
best_kept/    <- plain-file backup of the current best model.py/train.py (see below)
  model.py
  train.py
```

## `prepare.py`

Loads the official 100 training sequences (20 phases x 5 amplitudes) and
5 official test sequences (held-out phase per amplitude), groups the
training sequences into 5 folds by PHASE GROUP (never by amplitude --
see above), and builds initial-condition vectors. Each worker in
`train.py` loads its own fold's sequence data directly from disk rather
than receiving it via inter-process communication -- with 100 sequences
across 5 parallel workers, passing full tensor data through IPC exhausts
file descriptors; each worker reading only its own fold's files from
disk avoids this. **Should not be modified.**

## `model.py`

Contains a black-box recurrent baseline (`RNN`/`GRU`/`LSTM`) and a
grey-box parallel-branch model (`type: "PARALLELWH"`): two branches, each
`FIRBlock(n_taps_h) -> pointwise MLP -> FIRBlock(n_taps_s)`, matching the
true two-branch physical structure directly, with the branch outputs
summed. Fully vectorized (`conv1d`-based FIR blocks), no sequential
Python loop, no initial-condition estimation needed. **Can be modified.**

## `train.py`

Leave-one-fold-out CV, 5 parallel workers, each fold containing ~20
sequences (not a single sequence per fold -- use
`aggregate_metrics_across_sequences`-style handling for multi-sequence
folds, matching what this project was originally built with).
`config_pars` holds model/training hyperparameters (**can be modified**),
including optional per-experiment `fold_time_budget_seconds`/
`eval_every` overrides (default `None` -> use `prepare.py`'s shared
values). Decisive metric: `validation_rmse_norm_mean`.

## `test.py`

Evaluates the fold-checkpoint ensemble on the 5 official held-out-phase
test sequences (and the arrow test signal too, if present in your data --
see the note above), each amplitude-level result reported separately.
Monitoring only -- never used to choose between candidates.

## Typical Workflow

1. Read `ParallelWH_description.md`.
2. Run `prepare.py` (do not modify).
3. Modify `model.py` and/or `train.py`'s `config_pars`.
4. Run `train.py`, inspect `plots/cross_validation_curves.png` and the
   printed `validation_rmse_norm_mean`.
5. Optionally run `test.py` for monitoring only.
6. Repeat 3-5, logging to `results.tsv` and `search_journal.md`.
7. Once a winner is chosen, run `test.py --checkpoint-set best_so_far`
   once for the final reported numbers.

## Setting Up a New Search

No branch, no run tag. Read `ParallelWH_description.md` + `prepare.py` +
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
- The evaluation protocol (5-fold CV grouped by PHASE, the 5 official
  test sequences) must not be changed.

### Target Metric

Lowest possible `validation_rmse_norm_mean` (5-fold CV, normalized
units). The official per-sequence test RMSEs (from `test.py`) are for
reporting only.

## Search Strategy

Explore-then-refine, at least 15 total iterations:

1. First 6-7 runs: meaningfully different model families (black-box
   RNN/GRU/LSTM, grey-box `PARALLELWH`, other architectures if relevant).
2. Refine the most promising one through moderate hyperparameter changes.
3. If refinements plateau, try a different family instead.
4. Repeat until the 15-iteration minimum is met AND refinements have
   genuinely plateaued -- both conditions.

Notes specific to this benchmark:

- The two branches in `PARALLELWH` don't have to be identical in
  capacity (`n_taps_h`/`n_taps_s`, hidden-layer width per branch) -- if
  one branch is doing more of the work, an asymmetric configuration is
  worth trying explicitly rather than assuming symmetry.
- Since folds are grouped by phase (not amplitude), a candidate that
  does well on `validation_rmse_norm_mean` but shows a large per-
  amplitude spread when monitored via `test.py` is worth a closer look,
  even though that spread must never be used to choose between
  candidates directly (only `validation_rmse_norm_mean` can do that).
- `n_taps_h`/`n_taps_s` on `PARALLELWH`: if a candidate's fit looks
  capacity-limited, try larger values (the true system is 12th-order
  overall, split across the two branches) before concluding the FIR
  approach itself is unsuitable.

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
  already managed entirely by `train.py` itself via plain file copies --
  nothing to do here, this was never the part of the workflow that
  broke.
- The **very first (baseline) run** has nothing to back up yet -- after
  its `keep` decision (baseline is always initially kept, being the only
  candidate so far), just create the first `best_kept/` snapshot as
  above.

## Logging Results

`results.tsv`, tab-separated: `commit  val_RMSE  test_RMSE  status  description`.
Since there is no git, the `commit` column is not a git hash -- use a
short sequential label instead (e.g. `run01`, `run02`, ...) matching
`search_journal.md`'s own iteration numbering. `val_RMSE` =
`validation_rmse_norm_mean` (decisive); `test_RMSE` = the per-amplitude
RMSEs reported by `test.py` (semicolon-separated), logged every run
(monitoring only).

## Experiment Loop

1. First run: baseline, `model.py`/`train.py` unmodified. Always `keep`
   (only candidate so far) -- copy them into `best_kept/` (see above).
2. Every later run: modify `train.py`/`model.py` with one experimental
   idea.
3. Run `train.py`.
4. Inspect `plots/cross_validation_curves.png` and the printed
   `validation_rmse_norm_mean`.
5. Optionally run `test.py` for the monitoring-only per-amplitude RMSEs.
6. Decide `keep`/`discard` using `validation_rmse_norm_mean` only.
7. Update `results.tsv` and `search_journal.md`.
8. If `keep`: copy the current `model.py`/`train.py` into `best_kept/`.
   If `discard`: copy `best_kept/model.py`/`best_kept/train.py` back
   over the working files before the next experiment.
9. Do not conclude before at least 15 total iterations.
10. Once satisfied with a winner, run `test.py --checkpoint-set
    best_so_far` once and record that as the final reported numbers.

## Search Journal Rules

Explanatory entries (what changed, why, what happened, kept or discarded,
next step), not a terse ledger.

## AI Agent Rules

- Read `program.md` AND `ParallelWH_description.md` before starting. Do
  not assume any model family already wins -- reason from the physical
  description and let `validation_rmse_norm_mean` decide.
- Treat `prepare.py` as strictly read-only.
- **Never run any `git` command as part of this search** -- no commits,
  no branches, no checkouts. Use the plain-file `best_kept/` mechanism
  above for every keep/discard decision instead.
- Use `validation_rmse_norm_mean` (5-fold CV), and only that metric, to
  decide what to keep. `test.py` is for monitoring only.
- Update `results.tsv` and `search_journal.md` after each run.
- Do not conclude the search before at least 15 total iterations
  (baseline included) -- stopping early is premature regardless of how
  good the current winner looks.
- After every `keep`, copy `model.py`/`train.py` into `best_kept/`.
  After every `discard`, copy `best_kept/model.py`/`best_kept/train.py`
  back over the working files before continuing.
- Once satisfied, run `test.py --checkpoint-set best_so_far` once and
  record that single final result.
- Continue autonomously unless a real blocker requires user input.