# Bouc-Wen Hysteretic Benchmark

## 1. System Description

A single-DOF Bouc-Wen oscillator (Noël & Schoukens, 2016) -- a mechanical
system with **hysteresis**: the restoring force depends not just on the
instantaneous displacement/velocity, but on the system's own history
(the defining property of hysteresis: an input-output loop persists even
as the input frequency approaches zero, which linear systems cannot do).

```
mL*y'' + kL*y + cL*y' + z = u                                    (eq. 1-2)
z' = alpha*y' - beta*(gamma*|y'|*|z|^(nu-1)*z + delta*y'*|z|^nu)  (eq. 3)
```

`y`: displacement (the measured output). `u`: external force (input).
`z`: the hysteretic internal force -- **not directly measurable**, its
own first-order ODE (eq. 3) is what encodes the system's memory.

Physical parameters (paper's own Table 1, confirmed and reused directly):

| mL | cL | kL | alpha | beta | gamma | delta | nu |
|---|---|---|---|---|---|---|---|
| 2 | 10 | 5e4 | 5e4 | 1e3 | 0.8 | -1.1 | 1 |

Linear modal parameters: natural frequency 35.59 Hz, damping ratio 1.12%.
`fs = 750 Hz`.

## 2. Data: No Official Training Set, Real Official Test Set

**Confirmed directly from the paper's own Section 5**: *"The goal of the
benchmark is to estimate a good model on the estimation data... Two fixed
test datasets are provided through the benchmark meeting website."*
There is **no official training/estimation data file** -- participants
are expected to generate their own, following the paper's own recipe
(Sections 2-4). This project does exactly that (see Section 3 below).

The **two official test datasets ARE real and downloadable**, and this
project uses them directly (not self-generated, unlike an earlier version
of this project which approximated both). Verified directly against the
actual `.mat` files:

| Signal | Samples | Duration | Amplitude | Band | Steady-state? | Noise |
|---|---|---|---|---|---|---|
| multisine | 8192 | 10.9s | RMS 50N (verified) | 5-150Hz | yes | none |
| sine-sweep | 153000 | 204s | +-40N (verified) | 20-50Hz, 10Hz/min | no, starts at zero IC | none |

Both test sets are noiseless. **"The test data should not be used for any
purpose during the estimation"** (paper's own words) -- reported
separately, never combined, and never used to pick a winning candidate.

## 3. Self-Generated Training Data

Since no official file exists, `prepare.py` generates estimation data
following the paper's "minimal working example" recipe as closely as
possible:

- A random-phase multisine, band 5-150Hz, RMS 50N, `fs=750Hz`,
  `N=8192` samples/period.
- **5 periods simulated** (paper's own example), with **1 extra period
  prepended** to absorb the decimation filter's edge effects (paper's
  own guidance), then discarded.
- Integrated at **20x upsampling** (paper's own recommendation:
  `1/h = 15000 Hz`) via **RK4**, then decimated back to 750Hz by
  successive prime-factor `scipy.signal.decimate` calls (2-2-5, matching
  the paper's own suggested factorization of 20).
- **Not literal Newmark integration**: the paper's own scheme is
  implemented in an encrypted MATLAB p-file
  (`BoucWen_NewmarkIntegration.p`), unreadable and unusable from Python.
  RK4 is a standard, well-tested explicit alternative for this ODE (the
  `|.|` terms are continuous, just non-differentiable at `ydot=0`/`z=0`)
  -- verified directly: no NaN, physically sensible ~+-2.2mm displacement
  range, matching this benchmark's expected scale.
- Band-limited Gaussian output noise added, RMS 8e-3mm (paper's own
  example value) -- giving ~40dB SNR at the 50N excitation level, matching
  the paper. Input `u` is noiseless.

Split into **5 contiguous folds** for leave-one-fold-out CV (the same
"single shared recording, contiguous index-range folds" pattern used for
the CED/EMPS projects in this collection, since this is one continuous
generated trajectory, not independent realizations).

## 4. Numerical-Stability Fixes (carried forward from this collection's
established lessons)

- **`dt = Ts/num_substeps`, not `dt = 1/num_substeps`**: an earlier
  version of `BoucWenModel` used the latter, decoupling the integration
  step from the system's real timescale (`Ts = 1/750s`) entirely --
  found to cause severe numerical instability at every substep count
  tried (RK4 stability bound violated by 8-80x). Fixed by tying `dt`
  properly to the true sampling time.
- **Zero-initialized `state_init`'s output layer**: starts the model at
  a sensible `(y=0, ydot=0, z=0)` guess rather than an arbitrary random
  one (same fix used throughout this project collection).
- **Reference-scaled physical parameters**: `mL, cL, kL, alpha, beta`
  span 5 orders of magnitude (`cL=10` to `kL=5e4`) -- each is stored as
  a unit-scale multiplier of its Table-1 reference value, so a single
  learning rate treats all of them comparably (unlike raw, unscaled
  values, which would badly starve the smaller-magnitude parameters'
  gradients).
- **`nu` fixed at 1, not learned**: the paper's own Section 6 flags this
  exponent specifically as a harder-than-usual parameter ("the nonlinear
  functional form... is nonlinear in the parameter nu"). Fixing it at
  the paper's own true value avoids that specific difficulty while still
  exercising the rest of the hysteretic identification challenge (the
  internal, unmeasurable `z` state and its history-dependence).

## 5. Simulation Mode vs. Prediction Mode

**The benchmark explicitly asks for both** (paper's own Section 5):

- **Simulation**: `y_mod(t) = F(u(1..t))` -- driven only by the input,
  free-running/autoregressive. This is what every model in this project
  supports by default, and the ONLY mode used for `keep`/`discard`
  decisions (see `program.md`).
- **Prediction**: `y_mod(t) = F(u(1..t), y(1..t-1))` -- ALSO given the
  true past output at each step (one-step-ahead, teacher-forced).

`BoucWenRecurrentModel` supports BOTH via `use_output_feedback=True`
(default `False`, so existing behavior is unchanged unless explicitly
enabled): with output feedback on, the model takes `y(t-1)` as an
additional input at every step, and `test.py` automatically reports both
figures of merit for it. `BoucWenModel` (the grey-box physical model)
only supports simulation mode -- adding prediction-mode support to it
would mean correcting the FULL internal state (position, velocity, AND
the unmeasurable `z`) from a single observed `y`, which isn't
well-defined without additional assumptions; worth exploring as an
experiment if genuinely useful (e.g. via an observer/Kalman-style
correction), but not built in by default.

## 6. Nonlinear System Identification Challenges (paper's own Section 6)

1. A nonlinearity featuring **memory** (dynamic, not static).
2. Governed by an **internal variable `z(t)` that is not measurable**.
3. The functional form (eq. 3) is **nonlinear in the parameter `nu`**.
4. No finite Taylor series expansion exists, because of the `|.|` terms.

## 7. References

- J.P. Noël and M. Schoukens, *Hysteretic Benchmark With a Dynamic
  Nonlinearity*, Workshop on Nonlinear System Identification Benchmarks,
  Brussels, 2016 -- read directly for this project (both the standalone
  PDF and the copy bundled in the official zip, confirmed identical).
- Official data: the `BoucWenFiles.zip` (test signals + the encrypted
  Newmark integrator + example script) -- copy the extracted
  `BoucWenFiles/` folder next to `prepare.py` (see `prepare.py`'s own
  `RAW_MAT_DIR` comment for the exact expected layout).