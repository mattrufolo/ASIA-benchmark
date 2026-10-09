# Wiener-Hammerstein Benchmark (SYSID 2009) Program

This folder contains the material used to run autoresearch on the
classic Wiener-Hammerstein benchmark (Schoukens, Suykens & Ljung, 15th
IFAC Symposium on System Identification, Saint-Malo, France, 2009). The
goal is to train and evaluate models that describe the output voltage
from the input voltage for this real electronic circuit -- a SINGLE
Wiener-Hammerstein branch, measurement noise only (NOT the process-noise
variant, NOT the parallel two-branch variant -- see
`WienerHammerBenchMark_description.md` for how this differs from those).

## IMPORTANT: Read This Before Starting

**No git.** This project does not use git branches, commits, or
checkouts as part of the search loop -- see "Keeping and Discarding
Experiments" below for how `keep`/`discard` decisions are tracked
instead, using plain file copies confined to this folder. Do not run any
`git` command as part of this search, and do not create a branch for it.

**Minimum 15 total iterations before concluding the search.** This is a
hard floor, not a soft target -- do not stop early just because a result
looks conclusive.

**Data**: loaded via `nonlinear_benchmarks.WienerHammerBenchMark()`
(official Python loader, `github.com/MaartenSchoukens/nonlinear_benchmarks`).
Verified directly from the loader's own output: `train_val` is a single
100,000-sample recording, `test` is a separate 78,800-sample recording,
both at `sampling_time=1.953e-05s` (fs~=51200Hz), and the loader itself
specifies `state_initialization_window_length=50` -- this project uses
`history_window=50` to match exactly, so `warmup_test=0` on top of it.

```
step 1: SEARCH   train.py   -- leave-one-fold-out CV over 5 contiguous
                                folds of the single 100,000-sample
                                train_val recording (matches the
                                CED/Silverbox pattern in this collection:
                                one continuous recording, contiguous
                                index-range folds -- not independent
                                realizations)
step 2: TEST     test.py    -- evaluate the ENSEMBLE of the 5 fold checkpoints
                                on the official 78,800-sample test recording
```

Start from first principles, not assumed conclusions. No model family
should be assumed to win or lose ahead of time. Reason from
`WienerHammerBenchMark_description.md`'s physical description, then let
`validation_rmse_norm_mean` decide.

## Benchmark Description

Read
[WienerHammerBenchMark_description.md](./WienerHammerBenchMark_description.md)
before starting: it covers the physical `R(s) -> f(x) -> S(s)` structure
(diode-resistor nonlinearity between a 3rd-order Chebyshev input filter
and a 3rd-order inverse Chebyshev output filter with a transmission
zero -- confirmed directly from the benchmark's own papers), why there
is no process noise here (unlike the separate
`WienerHammersteinProcessNoise` project in this collection) and no
parallel branching (unlike `ParallelWH`), and the fold-design rationale.

## Folder Structure

```
prepare.py
model.py
train.py
test.py
WienerHammerBenchMark_description.md
best_kept/    <- plain-file backup of the current best model.py/train.py (see below)
  model.py
  train.py
```

## `prepare.py`

Downloads the official data via
`nonlinear_benchmarks.WienerHammerBenchMark()`, splits the single
100,000-sample `train_val` recording into 5 contiguous folds, builds
initial-condition vectors (`history_window = 50`, matching the loader's
own `state_initialization_window_length` exactly, so `warmup_test = 0`
on the official test set), and provides `sample_training_window()` for
random-crop training. **Should not be modified.**

## `model.py`

Contains a black-box recurrent baseline (`RNN`/`GRU`/`LSTM`) and a
Wiener-Hammerstein-style structural model (`type: "WIENERHAMMERSTEIN"`):
`R(s) -> f(x) -> S(s)`, each LTI block a learnable FIR filter (fully
vectorized `conv1d`, no sequential Python loop, no initial-condition
estimation needed at all), `f` a small pointwise MLP nonlinearity.
**Can be modified.**

## `train.py`

Leave-one-fold-out CV, 5 parallel workers. `config_pars` holds model/
training hyperparameters (**can be modified**), including optional
per-experiment `fold_time_budget_seconds`/`eval_every`/`device_override`
overrides (default `None` -> use `prepare.py`'s shared values). Decisive
metric: `validation_rmse_norm_mean`.

## `test.py`

Evaluates the fold-checkpoint ensemble on the official 78,800-sample
test recording:

```bash
python test.py
python test.py --checkpoint-set best_so_far
```

Reports RMSE in **Volts**. Monitoring only -- never used to choose
between candidates.

## Typical Workflow

1. Read `WienerHammerBenchMark_description.md`.
2. Run `prepare.py` (do not modify).
3. Modify `model.py` and/or `train.py`'s `config_pars`.
4. Run `train.py`, inspect `plots/cross_validation_curves.png` and the
   printed `validation_rmse_norm_mean`.
5. Optionally run `test.py` for monitoring only.
6. Repeat 3-5 for at least 15 total iterations, logging to `results.tsv`
   and `search_journal.md`.
7. Once a winner is chosen, run `test.py --checkpoint-set best_so_far`
   once for the final reported number.

## Setting Up a New Search

No branch, no run tag. Read `WienerHammerBenchMark_description.md` +
`prepare.py` + `model.py` + `train.py`. Initialize `results.tsv` (header
only) and `search_journal.md` (empty) if they don't already exist -- if
a previous search already left these populated and you're deliberately
starting a fresh search rather than continuing the existing one, clear
them first and say so explicitly in the first `search_journal.md` entry.
Create the `best_kept/` folder (empty is fine) if it doesn't exist.

## Experiment Rules

### What You Can Modify

- `model.py`
- `train.py` (`config_pars` and the training loop)

### What You Cannot Modify

- `prepare.py` -- read-only, no exceptions.
- No new dependencies.
- The evaluation protocol (5-fold CV over the single train_val
  recording, `history_window=50`, the official test recording) must not
  be changed.

### Target Metric

Lowest possible `validation_rmse_norm_mean` (5-fold CV, normalized
units). The official test RMSE (V, from `test.py`) is for reporting
only.

## Search Strategy

**Minimum 15-20 total iterations before concluding the search -- a hard
floor, not a soft target.** Do not stop early just because a candidate's
result looks conclusive.

Explore-then-refine:

1. *First 7-8 runs: meaningfully different model families (black-box
   RNN/GRU/LSTM, structural `WIENERHAMMERSTEIN`, some combination of the two, hybrids, or other architectures if
   relevant)*.
2. Refine the most promising one through moderate hyperparameter changes.
3. If refinements plateau, try a different family before concluding.
4. Repeat until the 15-iteration minimum is met AND refinements have
   genuinely plateaued -- both conditions, not just the first one
   reached.

Notes specific to this benchmark:

- The output filter `S(s)` has a transmission zero near 5kHz -- this
  makes the output dynamics genuinely hard to invert exactly; a
  candidate that plateaus above the noise floor isn't necessarily
  under-trained, this may be a real, expected difficulty of the
  benchmark itself.
- Since noise here is measurement-only (not process noise), a candidate
  that fits the training data very tightly is a much more directly
  interpretable signal of genuine progress than it would be on the
  process-noise variant -- there's no large irreducible noise floor to
  confuse with underfitting.
- `n_taps_r`/`n_taps_s` on `WIENERHAMMERSTEIN`: if a candidate's fit
  looks capacity-limited, try larger values before concluding the FIR
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
`validation_rmse_norm_mean` (decisive); `test_RMSE` = the RMSE (V)
reported by `test.py`, logged every run (monitoring only).

## Experiment Loop

1. First run: baseline, `model.py`/`train.py` unmodified. Always `keep`
   (only candidate so far) -- copy them into `best_kept/` (see above).
2. Every later run: modify `train.py`/`model.py` with one experimental
   idea.
3. Run `train.py`.
4. Inspect `plots/cross_validation_curves.png` and the printed
   `validation_rmse_norm_mean`.
5. Optionally run `test.py` for the monitoring-only RMSE.
6. Decide `keep`/`discard` using `validation_rmse_norm_mean` only.
7. Update `results.tsv` and `search_journal.md`.
8. If `keep`: copy the current `model.py`/`train.py` into `best_kept/`.
   If `discard`: copy `best_kept/model.py`/`best_kept/train.py` back
   over the working files before the next experiment.
9. Do not conclude before at least 15 total iterations.
10. Once satisfied with a winner, run `test.py --checkpoint-set
    best_so_far` once and record that as the final reported number.

## Search Journal Rules

Explanatory entries (what changed, why, what happened, kept or discarded,
next step), not a terse ledger.

## AI Agent Rules

- Read `program.md` AND `WienerHammerBenchMark_description.md` before
  starting. Do not assume any model family already wins -- reason from
  the physical description and let `validation_rmse_norm_mean` decide.
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
