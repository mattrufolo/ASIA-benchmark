# Silverbox Benchmark

## 1. System Description

An electronic implementation of a **Duffing oscillator**: a 2nd-order LTI
system with a 3rd-degree polynomial static nonlinearity in feedback
(package docstring). This type of dynamics is common in mechanical
systems -- the Silverbox circuit is a well-known electronic analogue used
as a nonlinear system identification benchmark since the mid-2000s
(originally presented at NOLCOS 2004, extended in Wigren & Schoukens,
ECC 2013).

```
                y'' + c*y' + k*y + k3*y^3 = u(t)
```

- `u(t)`: input voltage
- `y(t)`: output voltage
- Sampling: `fs = 610.35 Hz`
- Source file: `SNLS80mV.mat` (a second file, `Schroeder80mV.mat`, also
  ships with the download but isn't used in the default `train_test_split`
  mode)

**Note**: I don't currently have a Silverbox-specific technical report in
context (only the package source, its docstring, and the Wigren/Schoukens
citation) -- if you have the original benchmark document, it may be worth
sending so this section can be filled in more precisely (e.g. the true
`c`, `k`, `k3` values, if published).

## 2. Official Data (`nonlinear_benchmarks.Silverbox()`)

**Unique to this benchmark: there are THREE separate official test sets**,
evaluated and reported separately (never combined):

| Sequence | Excitation | Role |
|---|---|---|
| train (`multisine_train_val`) | multisine, first ~75% of the multisine recording | training |
| `test_multisine` | SAME multisine recording, held-out final ~25% -- a genuine continuation, like CED | test (in-distribution) |
| `test_arrow_full` | a different "arrow" sweep excitation, **including amplitudes beyond the training range** | test (extrapolation) |
| `test_arrow_no_extrapolation` | the same arrow excitation, trimmed to the trained amplitude range | test (in-range, different excitation type) |

`state_initialization_window_length = 50` for all three test sets.

This gives a genuinely richer evaluation than a single test set: comparing
`test_arrow_full` vs. `test_arrow_no_extrapolation` directly measures how
much a model degrades specifically from being asked to extrapolate beyond
what it saw in training -- separate from the (also real) challenge of
generalizing to a different excitation *type* (arrow vs. multisine) at
all.

## 3. Cross-Validation Design

The train recording and `test_multisine` are two slices of **one
continuous multisine recording** (like CED), so -- as with CED and
WienerHammerBenchMark -- contiguous folds of the single train recording
are used for leave-one-fold-out CV:

```
prepare.py:  split the train recording into 5 contiguous folds
train.py:    leave-one-fold-out CV across the 5 folds
test.py:     evaluate the fold-checkpoint ensemble on ALL THREE official
             test sets, reported SEPARATELY
```

`history_window = 50` matches the official test warmup exactly, so
`warmup_test = 0`. No `refit.py` -- same reasoning as the other projects.

## 4. Grey-Box Model in This Project

`SilverboxModel` (`model.py`) implements the actual physical structure
directly: a 2-state (`y`, `y'`) discrete-time recursion with a cubic
nonlinear term, reusing the exact same **discrete-time-native** pattern
already proven safe and fast on the CED and EMPS projects
(`WienerCEDModel`/`WienerEMPSModel`) -- no continuous-time `dt`, no RK4
sub-stepping.

This was a deliberate choice given the BoucWen project's own experience:
BoucWen's hysteretic state genuinely has no closed discrete form, forcing
a continuous-ODE-integrated-step-by-step approach that turned out fragile
(a `dt`-scaling bug caused persistent NaN) and slow (the resulting deep
sequential graph dominates backward-pass cost, and JIT scripting only
partially helped). The Duffing oscillator here needs no such thing -- a
simple 2-state recursion with one polynomial nonlinear term is exactly
the same shape of problem CED/EMPS already solved safely, so the same
proven pattern is reused rather than repeating BoucWen's heavier approach.

## 5. Why the Data Is Hard

- Genuine nonlinear resonance (the cubic term shifts the effective
  stiffness with amplitude -- classic Duffing "softening/hardening"
  behavior).
- The `test_arrow_full` extrapolation test directly probes whether a
  model's nonlinearity is well-identified (not just curve-fit within the
  training amplitude range) -- a model that overfits training-amplitude
  behavior specifically will likely show a large gap between
  `test_arrow_no_extrapolation` and `test_arrow_full`.
- Two different excitation *types* (multisine vs. arrow sweep) means some
  genuine extrapolation across excitation shape is unavoidable even
  within the "no extrapolation" test set.

## 6. References

- T. Wigren and J. Schoukens, *Three Free Data Sets for Development and
  Benchmarking in Nonlinear System Identification*, European Control
  Conference (ECC), pp. 2933-2938, Zurich, Switzerland, 2013.
- `nonlinear_benchmarks` package: https://github.com/MaartenSchoukens/nonlinear_benchmarks