# Parallel Wiener-Hammerstein Benchmark (ParallelWH)

## 1. System Description

A real electronic **2-branch parallel Wiener-Hammerstein** system
(Schoukens, Marconato, Pintelon, Vandersteen & Rolain, 2017): two
Wiener-Hammerstein branches sharing the same input, with outputs summed.

```
        +-- H[1](q) -> f[1](.) -> S[1](q) --+
u(k) -->|                                    +--> y(k) = sum of branch outputs
        +-- H[2](q) -> f[2](.) -> S[2](q) --+
```

- `H[i](q)`, `S[i](q)`: front/back LTI blocks of branch `i`, each a
  3rd-order continuous-time IIR filter in the real device.
- `f[i](.)`: a static nonlinearity per branch, realized with a
  diode-resistor network (same physical mechanism as
  WienerHammerBenchMark's single-branch nonlinearity, here duplicated per
  branch).
- Excitation: random-phase multisine, `N=131072` samples generated at
  625 kHz, band `[fs/N, 20 kHz]`.
- Measurement: 78 kHz (8x downsampled from the 625 kHz generator clock),
  giving 16384 measured samples per period; **2 periods are concatenated**
  per recorded sequence (32768 samples total).
- **5 excitation RMS levels**, linearly spaced from 100 mV to 1 V.
- **20 independent random-phase realizations** at each level, used for
  estimation (the paper needs >=4 realizations for a consistent BLA
  estimate; 20 gives a robust one).
- **1 held-out phase realization per level**, never used in estimation,
  for validation/test.

The paper's own measurement example uses exactly this configuration
(2 branches, 3rd-order front/back filters per branch, diode nonlinearity)
and reports the parallel WH model beating a plain single-branch
Wiener-Hammerstein model, a NARX model, and a nonlinear-output-error
model by roughly 10-20x in validation RMSE (Table 1 of the paper).

## 2. Official Data (`nonlinear_benchmarks.ParWH()`)

```
train: 100 sequences ("Est-phase-{0..19}-amp-{0..4}")
       = 20 phases x 5 amplitudes, 32768 samples each
test:  5 sequences ("Val-amp-{0..4}")
       = 1 held-out phase per amplitude, 32768 samples each
```

`state_initialization_window_length = 50` for every sequence. Report
units: Volts; the paper's own Table 1 reports validation error in
millivolts, broken out **per amplitude level** (never combined into one
number).

## 3. Why the Data Is Hard

- The static nonlinearity is sandwiched between two LTI blocks per
  branch, and there are TWO branches sharing the same input -- a
  genuinely harder identifiability problem than a single Wiener-Hammerstein
  branch (WienerHammerBenchMark): a gain/delay exchange between blocks
  within a branch is possible (same issue as single-branch WH), AND a
  full-rank linear transformation between the two branches' front/back
  dynamics is possible without changing the input-output behavior (see
  the paper's Section 3) -- the model can only be identified up to this
  degeneracy, not to the literal true per-branch parameters.
- 5 different excitation amplitudes exercise the nonlinearity differently
  (a static nonlinearity's effective local gain changes with signal
  amplitude) -- a model that only fits well at one amplitude level is
  likely missing the true nonlinear shape.
- The paper's own results show even a well-tuned single-branch
  Wiener-Hammerstein model, NARX, and NOE models all do noticeably worse
  (10-20x higher RMSE) than a properly-identified 2-branch model on this
  exact system -- strong independent evidence that the branch structure
  itself matters, not just capacity.

## 4. Cross-Validation Design (Fold Choice, Explained)

**This is the one important design decision worth spelling out.** The
official test task is: *generalize to an unseen phase realization, at
amplitude levels already seen in training* (all 5 RMS levels, 100 mV to
1 V, appear in both `Est-*` and `Val-*`). Given that, folding by
**amplitude level** would test the wrong thing entirely (whether the
model generalizes to an unseen *amplitude*, which the official test never
actually asks) -- the same class of fold/test mismatch that misled the
CED project's original design.

So folds here are built by **phase group**, not amplitude:

```
prepare.py:  20 phases split into 5 folds of 4 phases each; each fold
             contains ALL 5 amplitude levels for those 4 phases
             (4 x 5 = 20 sequences per fold, 100 total)
train.py:    leave-one-fold-out CV across the 5 phase-group folds --
             each validation fold is a held-out set of phases, but still
             spans the full amplitude range, matching the official test
             task exactly
test.py:     evaluate the fold-checkpoint ensemble on all 5 official
             test sequences, reported SEPARATELY per amplitude level
```

Each fold is a **list of 20 independent sequences**, not one array (unlike
CED/EMPS/WienerHammerBenchMark's contiguous-recording folds) -- these are
genuinely independent phase/amplitude realizations, so training samples a
random window from a randomly-chosen sequence in the available pool each
step (the same pattern used for BoucWen's independent-realization folds).

No `refit.py`: with 100 total training sequences (~80 per fold), holding
one fold back is not a meaningful data sacrifice.

## 5. Grey-Box Model in This Project

`ParallelWHModel` (`model.py`) implements the true 2-branch structure
directly: each branch is a learnable causal **FIR** filter (`H[i]`) ->
small pointwise MLP (`f[i]`) -> another learnable causal FIR filter
(`S[i]`), with branch outputs summed. This reuses the FIR-block design
from the WienerHammerBenchMark project (fully vectorized `conv1d`, no
sequential Python loop, no initial-condition estimation needed at all)
rather than a recursive IIR/state-space realization -- deliberately,
given the BoucWen/Silverbox projects' experience that a sequential
per-timestep recursion for a genuinely nonlinear system is both fragile
(numerical instability issues needing careful sign/saturation fixes) and
slow to backpropagate through.

`n_branches` defaults to 2 (matching the real device), but is
configurable if a different branch count is worth testing.
`n_taps_h`/`n_taps_s` trade FIR approximation quality for the true
3rd-order IIR responses against model size/training cost.

## 6. References

- M. Schoukens, A. Marconato, R. Pintelon, G. Vandersteen, and Y. Rolain,
  *Parametric Identification of Parallel Wiener-Hammerstein Systems*,
  Automatica, 2017 (arXiv:1708.06543).
- TU/e dataset page: https://research.tue.nl/en/datasets/parallel-wiener-hammerstein-time-series/
- `nonlinear_benchmarks` package: https://github.com/MaartenSchoukens/nonlinear_benchmarks