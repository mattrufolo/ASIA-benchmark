# Coupled Electric Drives (CED) Benchmark

## 1. System Description

The Coupled Electric Drives (CE8) process consists of two electric motors
that drive a pulley through a flexible belt. The pulley is held by a spring,
which produces a lightly damped resonant mode. The two drives are controlled
symmetrically, so the belt can rotate in either direction; the pulley speed
is measured with a pulse counter whose sensor is insensitive to the sign of
the velocity (it only reports a rectified, always-positive speed).

```
u(t) -> [combined motor voltage] -> drives + spring/belt -> pulley -> pulse counter -> y(t)
```

- Input: `u(t)`, the sum of the voltages applied to the two motors [V]
- Output: `y(t)`, the rectified pulley speed [ticks/s]
- Sampling time: `Ts = 0.02 s` (50 Hz)

---

## 2. Physical Modeling (Grey-box, Wiener structure)

The technical report (Wigren & Schoukens, 2017) models the drives as a
lightly damped, third-order **linear** system followed by a **static
output rectifier** — i.e. a Wiener model:

```
          k * alpha * omega_0^2
y(s) = ------------------------------ u(s)          (linear part)
        (s + alpha)(s^2 + 2*xi*omega_0*s + omega_0^2)

y_n(t) = |y(t)| + e(t)                                (static output nonlinearity)
```

- `alpha`: inverse time constant of the electric drives
- `xi`, `omega_0`: damping and resonance frequency of the spring/belt mode
- `k`: static gain
- `e(t)`: measurement disturbance

An equivalent continuous-time state-space (companion) form used for
recursive identification in the report is:

```
x1' = x2
x2' = x3
x3' = theta_1*u + theta_2*x3 + theta_3*x2 + theta_4*x1
y   = |x1|
```

This has only 4 free parameters (vs. 6 for an equivalent discrete-time
model), and the paper additionally proposes a **Wiener–Hammerstein**
extension that adds an unknown linear anti-aliasing filter (bandwidth
~12 Hz) after the rectifier:

```
z(t)  = |y(t)|
y_n(s) = (b_1 s^{n-1} + ... + b_n) / (s^n + a_1 s^{n-1} + ... + a_n) * z(s) + v(s)
```

The report notes the resonance peak sits around 25 rad/s (~4-5 Hz) and
the drive bandwidth is a few rad/s. The system is a de-facto standard for
nonlinear system identification (soft rectification nonlinearity + light
damping + very short data records).

---

## 3. Official Benchmark Data (`nonlinear_benchmarks.CED()`)

**Important**: the official benchmark leaderboard split — the one this
project reproduces — only uses the **uniformly-distributed-amplitude**
input realizations from the technical report (`DATAUNIF.MAT`, Section 2.2
of the report), **not** the three PRBS-only realizations (`DATAPRBS.MAT`,
Section 2.1). There are exactly two realizations:

| Realization | Input switching levels | Samples | Role |
|---|---|---|---|
| `low_amplitude`  (u11/z11) | -1.5 V / +2.5 V, scaled uniform in [0,1] | 500 | train[0:400], test[400:500] |
| `high_amplitude` (u12/z12) | -1.0 V / +3.0 V, scaled uniform in [0,1] | 500 | train[0:400], test[400:500] |

- `Ts = 0.02 s` for both realizations.
- The official test split provides `state_initialization_window_length = 10`:
  the first 10 samples of each 100-sample test segment may be used to
  initialize the model's state, and RMSE is only scored on the remaining
  90 samples.
- Reporting units: RMSE in **ticks/s**, reported as a pair
  `[test_low_RMSE; test_high_RMSE]` (see
  `nonlinear_benchmarks/submission_examples/CED.py`).

---

## 4. Why the Data Is Hard

- Very short records: only 400 usable training samples per realization
  (800 total across both), and both realizations are the *only* source of
  training data — there is no third/fourth held-out realization to spare.
- A hard/soft nonlinearity at the output: full-wave rectification `|y|`
  folds negative and positive velocities on top of each other, which is a
  much harder inductive bias to learn than a smooth saturation.
- Lightly damped resonance (`xi`, `omega_0`) close to the Nyquist range of
  the 50 Hz sampling, so aliasing/ringing artifacts are easy to overfit to
  in such a small dataset.
- The two realizations differ in input amplitude/switching levels, which
  changes the operating point and, given the nonlinearity, the qualitative
  shape of the response — a model that only sees one regime at training
  time tends to generalize poorly to the other.

---

## 5. Search / Refit / Test Procedure Used in This Project

An earlier version of this project used 4-fold leave-one-fold-out
cross-validation (2 realizations x 2 contiguous halves each). That design
was abandoned: the official test window `[400:500]` is a visibly calmer,
lower-amplitude continuation of the recording, qualitatively different from
the busy middle section of `[0:400]` (see `plots/*_train_trajectory.png`).
None of the 4 interior folds ever validated a model's ability to
extrapolate to that kind of calm continuation, so architecture choices
selected by 4-fold CV systematically **overfit the CV metric at the
expense of true test performance** — every non-baseline architecture that
improved 4-fold validation RMSE made the official test RMSE noticeably
*worse*.

The procedure now used has three explicit steps:

```
step 1: SEARCH   train on [10:350] (both regimes), validate on [350:400] (both regimes)
step 2: REFIT    retrain the SAME config on [10:400]   (both regimes, no held-out split)
step 3: TEST     evaluate the refit model ONCE on the official [400:500]  (both regimes)
```

- **Step 1 (search)**: `train.py` trains one model jointly on the two
  400-sample train realizations, truncated to `[history_window:350]`, and
  validates on the held-out `[350:400]` window of both realizations. This
  validation window sits immediately before, and is qualitatively similar
  to, the official test window — unlike an interior CV fold. **All
  architecture and hyperparameter decisions are made using this
  search-validation RMSE, and only this metric.**
- **Step 2 (refit)**: once a winning configuration has been chosen,
  `refit.py` retrains that *same* configuration on the full
  `[history_window:400]` of both realizations (no held-out split — the
  `[350:400]` window is now training data). There is no validation signal
  at this stage, so training runs for a fixed number of epochs derived from
  the step-1 run's best epoch, rather than early-stopping.
- **Step 3 (test)**: `test.py` evaluates the refit checkpoint once on the
  official `[400:500]` test split. This number is for reporting only.

**A hard rule that must never be violated**: the `[350:400]` window can be
used to *decide between models* (step 1) **or** used as *training data for
the final model* (step 2) — never both. Once a config is chosen and
refit on the full data, the `[350:400]` window must not be re-evaluated to
reconsider that choice; doing so would be double-dipping into the same
data for both selection and training, which reintroduces exactly the
optimistic bias the new procedure is meant to avoid.

`history_window = 10` is unchanged from before, and for the same reason:
it matches the official test warmup exactly, so `warmup_test = 0` and the
full 90 remaining test samples per realization are scored.


---

## 6. References

- T. Wigren and M. Schoukens, *Coupled Electric Drives Data Set and
  Reference Models*, Technical Report 2017-024, Dept. of Information
  Technology, Uppsala University, 2017.
- `nonlinear_benchmarks` package: https://github.com/MaartenSchoukens/nonlinear_benchmarks