# Silverbox Benchmark Program

This folder contains the material used to run autoresearch on the
Silverbox benchmark. The goal is to train and evaluate models that
describe the output voltage from the input voltage.

## IMPORTANT: Read This Before Starting

**Three separate official test sets exist** (`test_multisine`,
`test_arrow_full`, `test_arrow_no_extrapolation`), evaluated and reported
SEPARATELY -- never combined. `test_arrow_full` specifically includes
amplitudes beyond the training range (a genuine extrapolation test); see
`Silverbox_description.md` Section 2 for the full breakdown.

Train and `test_multisine` are two slices of one continuous multisine
recording (like CED), so contiguous folds of the single train recording
are used for leave-one-fold-out CV, and there is no separate refit step:

```
step 1: SEARCH   train.py   -- leave-one-fold-out CV over 5 contiguous folds
step 2: TEST     test.py    -- evaluate the ENSEMBLE of the 5 fold checkpoints
                                on ALL THREE official test sets, RMSE in V,
                                reported separately
```

**Start from first principles, not assumed conclusions.** No model family
should be assumed to win or lose ahead of time. Reason from
`Silverbox_description.md`'s physical description and general
system-identification/ML principles, then let `validation_rmse_norm_mean`
decide.

## Benchmark Description

Read [Silverbox_description.md](./Silverbox_description.md) before
starting: it covers the physical Duffing-oscillator structure, the
three-test-set data design, and why this project's grey-box model reuses
the CED/EMPS discrete-time-native pattern rather than BoucWen's heavier
continuous-ODE approach.

## Folder Structure

- `prepare.py`
- `model.py`
- `train.py`
- `test.py`
- `Silverbox_description.md`

## `prepare.py`

Downloads the official data via `nonlinear_benchmarks.Silverbox()`,
splits the train recording into 5 contiguous folds, builds
initial-condition vectors (`history_window = 50`, matching the official
test warmup so `warmup_test = 0`), and provides `sample_training_window()`
for random-crop training. **Should not be modified.**

## `model.py`

Contains a black-box recurrent baseline (`RNN`/`GRU`/`LSTM`) and
`SilverboxModel` (`type: "SILVERBOX"`): a 2-state discrete-time recursion
with a cubic nonlinear term, matching the true Duffing-oscillator
structure. **Can be modified.**

## `train.py`

Leave-one-fold-out CV, 5 parallel workers. `config_pars` holds model/
training hyperparameters (**can be modified**), including optional
per-experiment `fold_time_budget_seconds`/`eval_every` overrides (default
`None` -> use `prepare.py`'s shared values). Decisive metric:
`validation_rmse_norm_mean`.

## `test.py`

Evaluates the fold-checkpoint ensemble on all three official test sets:

```bash
python test.py
python test.py --checkpoint-set best_so_far
```

Reports RMSE in **Volts**, separately for each of the three test sets.
Monitoring only -- never used to choose between candidates. Worth
inspecting the *gap* between `test_arrow_no_extrapolation` and
`test_arrow_full` specifically -- a large gap suggests a candidate is
overfitting the training amplitude range rather than genuinely
identifying the nonlinearity.

## Typical Workflow

1. Read `Silverbox_description.md`.
2. Run `prepare.py` (do not modify).
3. Modify `model.py` and/or `train.py`'s `config_pars`.
4. Run `train.py`, inspect `plots/cross_validation_curves.png` and the
   printed `validation_rmse_norm_mean`.
5. Optionally run `test.py` for monitoring only.
6. Repeat 3-5, logging to `results.tsv` and `search_journal.md`.
7. Once a winner is chosen, run `test.py --checkpoint-set best_so_far`
   once for the final reported numbers (all three test sets).



## Experiment Rules

### What You Can Modify

- `model.py`
- `train.py` (`config_pars` and the training loop)

### What You Cannot Modify

- `prepare.py` -- read-only.
- No new dependencies.
- The evaluation protocol (5-fold split, `history_window=50`, the three
  official test sets) must not be changed.

### Target Metric

Lowest possible `validation_rmse_norm_mean` (5-fold CV, normalized
units). The official test RMSEs (V, from `test.py`, all three sets) are
for reporting only.

## Search Strategy
**Minimum 15 total iterations before concluding the search** -- this is
a hard floor, not a soft target.

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

- `SilverboxModel`'s `a_init` is a discrete-time pole and must stay
  strictly inside the unit circle (enforced at construction); if a
  candidate's training loss freezes near its untrained value from epoch
  0, suspect an unstable pole initialization or learning-rate issue
  before concluding the architecture is unsuitable.
- Given the three-test-set design, a candidate that looks good on
  `validation_rmse_norm_mean` but shows a large gap between
  `test_arrow_no_extrapolation` and `test_arrow_full` when monitored is
  worth a closer look at, even though the gap itself must never be used
  to choose between candidates directly (only `validation_rmse_norm_mean`
  can do that).
- `state_init`'s output layer is zero-initialized in `SilverboxModel`
  (starts at rest, `y=y'=0`) -- if experimenting with a different
  initialization scheme, keep in mind the BoucWen project's finding that
  an untrained network's wild-scale random initial state can destabilize
  even an otherwise well-conditioned recursion.

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
`val_RMSE` = `validation_rmse_norm_mean` (decisive); `test_RMSE` = record
as `multisine=X.XXX;arrow_full=Y.YYY;arrow_no_extrap=Z.ZZZ` (V) from
`test.py`, logged every run (monitoring only). `results.tsv` is a local
log, untracked by Git.

## Experiment Loop

Baseline first, one experimental idea per commit, run `train.py`, read
`validation_rmse_norm_mean`, decide `keep`/`discard`, update
`results.tsv`/`search_journal.md`, reset to best-kept commit on `discard`,
finish with one `test.py --checkpoint-set best_so_far` pass once a winner
is chosen.

## Search Journal Rules

Explanatory entries (what changed, why, what happened, kept or discarded,
next step), not a terse ledger. Start fresh at iteration 0 for each new
run tag.

## AI Agent Rules


- read `program.md` before starting
- Treat `prepare.py` as read-only. Modify only `model.py` and `train.py`.
- modify only `model.py` and `train.py`
- **never run any `git` command as part of this search** -- no commits,
  no branches, no checkouts. Use the plain-file `best_kept/` mechanism
  above for every keep/discard decision instead
- Use `validation_rmse_norm_mean` (5-fold CV), and only that metric, to
  decide what to keep. `test.py` (all three test sets) is for monitoring
  only.
- Update `results.tsv` and `search_journal.md` after each run.
- Once satisfied, run `test.py --checkpoint-set best_so_far` once and
  record that single final result (all three test sets).
- use `logs/train_search.log` to inspect training behavior
- update `results.tsv` after each completed SEARCH run 
- update `search_journal.md` after each completed run
- after every `keep`, copy `model.py`/`train.py` into `best_kept/`
- after every `discard`, copy `best_kept/model.py`/`best_kept/train.py`
  back over the working files before continuing
- continue autonomously unless a real blocker requires user input