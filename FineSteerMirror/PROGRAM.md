# CubeSpec Fine Steering Mirror (FSM) Benchmark Program

This folder contains the material used to run autoresearch on the FSM
benchmark. The goal is to train and evaluate models that describe 3
mirror displacement signals from 3 piezo-actuator voltage inputs (a
genuine 3-in, 3-out MIMO system).

## IMPORTANT: Read This Before Starting

**No git.** This project does not use git branches, commits, or
checkouts as part of the search loop -- see "Keeping and Discarding
Experiments" below. Do not run any `git` command as part of this
search, and do not create a branch for it.

**Minimum 15 total iterations before concluding the search.**

**Data setup**: `prepare.py` instead
reads directly from the `fsm-benchmark-data` repo's own `data/` folder
(github.com/merijnfloren/fsm-benchmark-data) -- copy that repo's `data/`
folder into this project's own directory, so
`fsm-benchmark-data/data/u_100mV_train.npy` etc. exist (see
`prepare.py`'s own `RAW_DATA_LOCAL_DIR` comment). `fs=6400Hz` is
hardcoded (confirmed directly from the paper, since it isn't embedded as
metadata in these files); `history_window=100` is a reasoned default,
not an authoritative package-provided value -- if the official loader
becomes available later, prefer its own `state_initialization_window_
length` over this default.

```
step 1: SEARCH   train.py   -- 6-fold leave-one-realization-out CV
                                (each fold holds out 1 of 6 realizations
                                from ALL 3 amplitude levels at once, 3
                                sequences held out per fold)
step 2: TEST     test.py    -- evaluate the ENSEMBLE of the 6 fold checkpoints
                                on the 3 OFFICIAL test amplitude levels,
                                per-output RMSE in micrometers, reported
                                separately per level
```

Start from first principles, not assumed conclusions. No model family
should be assumed to win or lose ahead of time. Reason from
`FSM_description.md`'s physical description, then let
`validation_rmse_norm_mean` decide.

## Benchmark Description

Read [FSM_description.md](./FSM_description.md) before starting: the
3-in/3-out MIMO structure, the confirmed piezo-actuator hysteresis
nonlinearity (grounded directly in the actual paper and presentation),
the data's period/realization tensor structure and how this project
reshapes it into sequences, the fold design, and the white-box model's
own physical justification.

## Folder Structure

```
prepare.py
model.py
train.py
test.py
FSM_description.md
fsm-benchmark-data/data/    <- you place this (see Data setup above)
best_kept/                   <- plain-file backup of the current best model.py/train.py
  model.py
  train.py
```

## `prepare.py`

Loads data directly from `fsm-benchmark-data/data/*.npy` (see Data setup
above). The raw `(N, 3, R, P)` tensors (P=2 steady-state periods, R=6
train/3 test realizations, 3 amplitude levels) are reshaped into
per-realization sequences by concatenating the 2 periods (the excitation
is genuinely periodic/continuous across the period boundary) -- 6
training sequences
+ 3 official test sequences per amplitude level, named
`level{01,02,03}_r{01..06}` / `test_level{01,02,03}_r{01..03}`. Input is
3 actuator voltages; output is all 3 displacements jointly.
`history_window=100` is a reasoned default (see Data setup above --
the official `state_initialization_window_length` isn't available since
we're reading local files, not the package). Read-only.

## `model.py`

Contains a black-box MIMO recurrent baseline (`RNN`/`GRU`/`LSTM`) and
`HYSTERESIS_LFR`: 3 independent Bouc-Wen-style hysteresis operators (one
per piezo-actuator, matching the confirmed nonlinearity mechanism) feed
a bank of stable discrete-time modes -- both oscillatory (`n_complex_
modes`, default 3, frequencies spread across the paper's own confirmed
750-950Hz dominant peak band) and real/non-oscillatory (`n_real_modes`,
default 2, directly matching the paper's own confirmed 2-real-pole
finding in its identified linear model). Outputs are a learnable linear
combination of all modal states. Can be modified -- other architectures
(TCN, attention, more/fewer modes, coupled rather than independent
hysteresis) are welcome.

**If touching the modal recursion's discretization**: this project's
`a_coef`/`b_coef` formula was derived specifically for a semi-implicit
(update-velocity-then-position) Euler step -- verified directly that
reusing the *differently-derived* formula from this collection's
Silverbox/F16 projects is only stable at low frequencies and genuinely
unstable above ~200Hz at the default `e_pos`; this benchmark's confirmed
750-950Hz dominant peaks would have silently hit that bug. Re-derived
and verified stable up to and beyond `fmax=3000Hz` at the real
`fs=6400Hz`. If you add a new physics-integrated model, re-derive and
numerically verify its own stability rather than assuming a formula
transfers from elsewhere in this collection, even one that "worked" on a
related model before.

## `train.py`

6-fold leave-one-realization-out CV, 6 parallel workers, each fold's
held-out set = 3 sequences (one per amplitude level). `config_pars`
holds model/training hyperparameters (can be modified), including
optional per-experiment `fold_time_budget_seconds`/`eval_every`/
`device_override` overrides. Windowed training from the start (sequences
are ~16k samples). Decisive metric: `validation_rmse_norm_mean`.

## `test.py`

Evaluates the fold-checkpoint ensemble on the 3 official amplitude
levels:

```bash
python test.py
python test.py --checkpoint-set best_so_far
```

Reports RMSE in **micrometers**, per output channel AND a combined
number per amplitude level (matching
`nonlinear_benchmarks/submission_examples/FineSteeringMirror.py`'s own
convention, confirmed directly this session), plus NRMSE (%). Monitoring
only -- never used to choose between candidates.

## Typical Workflow

1. Read `FSM_description.md`.
2. Run `prepare.py` (do not modify) -- reads from the local
   `fsm-benchmark-data/data/` folder (see Data setup above).
3. Modify `model.py` and/or `train.py`'s `config_pars`.
4. Run `train.py`, inspect `plots/cross_validation_curves.png` and the
   printed `validation_rmse_norm_mean`.
5. Optionally run `test.py` for monitoring only.
6. Repeat 3-5 for at least 15 total iterations, logging to `results.tsv`
   and `search_journal.md`.
7. Once a winner is chosen, run `test.py --checkpoint-set best_so_far`
   once for the final reported numbers.

## Setting Up a New Search

No branch, no run tag. Read `FSM_description.md` + `prepare.py` +
`model.py` + `train.py`. Initialize `results.tsv` (header only) and
`search_journal.md` (empty) if they don't already exist -- if continuing
from a stale/incompatible earlier run, clear them and say so explicitly
in the first journal entry. Create `best_kept/` if it doesn't exist.

## Experiment Rules

### What You Can Modify

- `model.py`
- `train.py` (`config_pars` and the training loop)

### What You Cannot Modify

- `prepare.py` -- read-only, no exceptions.
- No new dependencies.
- The evaluation protocol (6-fold leave-one-realization-out CV, the
  period-concatenation sequence design, the 3 official test levels)
  must not be changed.

### Target Metric

Lowest possible `validation_rmse_norm_mean` (6-fold CV, normalized
units, all 3 output channels). The official per-level per-output test
RMSEs (from `test.py`, µm) are for reporting only.

## Search Strategy

**Minimum 15 total iterations before concluding the search -- a hard
floor, not a soft target.**

Explore-then-refine:

1. First 6-7 runs: meaningfully different model families (black-box
   RNN/GRU/LSTM, `HYSTERESIS_LFR`, other architectures if relevant).
2. Refine the most promising one through moderate hyperparameter changes.
3. If refinements plateau, try a different family before concluding.
4. Repeat until the 15-iteration minimum is met AND refinements have
   genuinely plateaued.

Notes specific to this benchmark:

- This is a genuine 3-input, 3-output MIMO system -- don't assume
  intuitions from this collection's single-input benchmarks transfer
  directly (e.g. per-channel normalization, per-channel gradient scale,
  and cross-channel coupling all matter more here).
- `HYSTERESIS_LFR`: if training loss freezes near its untrained value
  from epoch 0, suspect the same class of bug already found (and fixed)
  in this collection's Silverbox/CED/F16 white-box models -- an
  unconstrained/badly-scaled pole, or a discretization-order mismatch
  between the derived stability formula and the actual recursion (see
  the `model.py` note above) -- before concluding the architecture is
  unsuitable.
- Since the 3 amplitude levels are physically the SAME actuators at
  different drive levels, a candidate that generalizes across levels
  (not just within one) is testing something physically meaningful, not
  an artifact of the fold design -- but see the next point before
  over-weighting this.
- **Confirmed directly from the paper**: much of the poor cross-level
  generalization the original authors found is hypothesized to be a
  measurement-campaign artifact (amplifier/generator behavior shifting
  slightly with drive level), not necessarily true system nonlinearity.
  Don't assume a candidate that generalizes poorly across levels is
  wrong, or that a large validation-RMSE gap between levels is only
  fixable by a better nonlinearity model -- it may be reflecting a
  genuine property of the data, not a modeling shortfall.
- `n_complex_modes`/`n_real_modes` on `HYSTERESIS_LFR`: the paper's own
  identified linear model is 28th-order overall and contains 2 real
  poles specifically -- `n_real_modes=2` is not an arbitrary default.
  `n_complex_modes`' frequencies default to a spread across the paper's
  own confirmed dominant peak band (750-950Hz) -- still worth sweeping
  both counts (this project's default total order, 3 complex x 2 + 2
  real = 8, is far below the paper's 28, deliberately: a reasonable,
  cheap starting point for the search to build up from, not a claim
  that 8 states suffice).

## Keeping and Discarding Experiments (no git)

**Everything here is a plain file copy, confined to this project's own
folder. No git command is ever used.**

- **After `keep`**: `cp model.py best_kept/model.py`,
  `cp train.py best_kept/train.py`.
- **After `discard`**: `cp best_kept/model.py model.py`,
  `cp best_kept/train.py train.py`.
- Checkpoints are already managed by `train.py` via plain file copies.
- First (baseline) run: after its `keep`, create the first `best_kept/`
  snapshot.

## Logging Results

`results.tsv`, tab-separated: `commit  val_RMSE  test_RMSE  status  description`.
`commit`: sequential label (`run01`, `run02`, ...), no git hash.
`val_RMSE`: `validation_rmse_norm_mean` -- decisive. `test_RMSE`: record
as `overall_mean_um=X` (from `test.py`'s summary), logged every run
(monitoring only).

## Experiment Loop

1. First run: baseline, unmodified -- always `keep`, snapshot to
   `best_kept/`.
2. Later runs: one experimental idea at a time.
3. Run `train.py`.
4. Inspect `plots/cross_validation_curves.png`, printed
   `validation_rmse_norm_mean`.
5. Decide `keep`/`discard` on `validation_rmse_norm_mean` alone.
6. Update `results.tsv`/`search_journal.md`; copy to/from `best_kept/`.
7. Continue until 15+ iterations and refinements have plateaued.
8. Once satisfied: `test.py --checkpoint-set best_so_far` once, record
   the final row.

## Search Journal Rules

Explanatory entries (what changed, why, what happened, kept/discarded,
next step), not a terse ledger.

## AI Agent Rules

- Read `program.md` + `FSM_description.md` before starting.
- Treat `prepare.py` as strictly read-only.
- **Never run any `git` command** -- use `best_kept/` instead.
- Use `validation_rmse_norm_mean`, and only that, to decide what to keep.
- Update `results.tsv`/`search_journal.md` after each run.
- Do not conclude before 15+ iterations.
- After `keep`/`discard`, update `best_kept/` accordingly.
- Once satisfied: `test.py --checkpoint-set best_so_far` once, record
  the final result.
- Continue autonomously unless a real blocker needs user input.