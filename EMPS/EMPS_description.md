# EMPS Benchmark (Electro-Mechanical Positioning System)

Summary of: A. Janot, M. Gautier, M. Brunot, *"Data Set and Reference Models of EMPS"*,
Workshop on Nonlinear System Identification Benchmarks, Eindhoven, 2019.

## 1. System Description

The EMPS is a standard drive system for a prismatic joint of a robot or a machine tool.
Physically it is made of:

- a Maxon DC motor with an incremental encoder, position-controlled with a PD controller,
- a high-precision, low-friction ball-screw drive positioning unit,
- a load in translation.

An encoder at the tip of the screw and an accelerometer on the load are mounted on the
prototype but **not used** in this benchmark, because such measurements are not available
on most industrial robots.

### Structure

```
vir(t) --> [ball-screw + DC motor drive] --> qm(t)
```

- Input: `vir(t)`, the motor force referred to the load side, in N.
- Output: `qm(t)`, the joint (load) position, in m.
- The system has **one degree of freedom**.

## 2. Physical Modeling (White-box)

### Inverse Dynamic Model (IDM)

From Newton's law, the joint torque/force is:

```
tau(t) = M*qdd(t) + Fv*qd(t) + Fc*sign(qd(t)) + offset
```

- `M`: inertia/mass of the arm
- `Fv`: viscous friction coefficient
- `Fc`: Coulomb friction coefficient
- `offset`: measurement offset

`M`, `Fv`, `Fc`, `offset` are the **base parameters** (structurally identifiable).

### Direct Dynamic Model (DDM)

Solving for the acceleration gives the simulation form used for benchmarking:

```
qdd(t) = tau(t)/M - (Fv/M)*qd(t) - (Fc/M)*sign(qd(t)) - offset/M
```

Because `vir(t)` is already expressed in the load side (i.e., it plays the role of
`tau(t)` in this benchmark), this is a second-order, control-affine ODE in the state
`[q, qd]` driven directly by the measured force `vir(t)`.

### Asymmetric friction

Later work (Janot et al., 2017) showed that the friction is in fact **asymmetric**,
attributed to mechanical fatigue of the screw:

```
tau_fric = Fv+ * 0+(qd) + Fc+ * sign(0+(qd)) + Fv- * 0-(qd) + Fc- * sign(0-(qd))
```

where `0+(qd) = qd*(1+sign(qd))/2` and `0-(qd) = qd*(1-sign(qd))/2` isolate the
positive/negative-velocity branches. This asymmetric model measurably improves the fit
over the symmetric one.

## 3. Control

The EMPS has a **pure integrator** and cannot be identified in open loop. It is driven by
a nested PD controller:

```
nu(t) = kp*kv*(qr(t) - q(t)) - kv*qd(t)
tau(t) = g_tau * nu(t)
```

with `kp = 160.18 1/s`, `kv = 243.45 V/(m/s)`, `g_tau = 35.15 N/V`. Because the controller
has two nested loops, it cannot be written as a single transfer function
`nu(t) = C(s)*(qref(t) - q(t))` as is usually assumed by generic identification
toolboxes — this is one of the reasons standard black-box toolboxes perform poorly on
this benchmark (relative errors of 40-75% reported in the paper) unless the *whole
closed loop* (`qg -> qm`) is modeled directly.

## 4. Data Set

- Sampling frequency: 1 kHz (`dt = 1 ms`), duration ≈ 25 s per trajectory.
- Excitation: bang-bang (piecewise-constant) accelerations, chosen because they excite
  inertia while acceleration varies and excite friction while velocity is constant.
- Provided signals: `qm` (position, m), `qg` (position reference, m), `vir` (force, N),
  `t` (time, s), plus the controller/drive gains `kp`, `kv`, `gtau`.
- **Input** for identification: `vir`. **Output**: `qm`. Data are raw/untreated.
- A second trajectory (`DATA_EMPS_PULSES`, pulses superimposed on bang-bang
  accelerations) is reserved for cross-test validation / official test set.
- Condition number of the observation matrix ≈ 26-30 → parameters are considered well
  excited.

In the `nonlinear_benchmarks` Python package, `nonlinear_benchmarks.EMPS()` returns
exactly these two trajectories (`train`, `test`), with `test.state_initialization_window_length = 20`
samples reserved for state initialization on the test trajectory (not scored).

## 5. Identification Goals

- Recover the physical parameters `M`, `Fv`, `Fc`, `offset` (and, if using the asymmetric
  friction model, `Fv+`, `Fc+`, `Fv-`, `Fc-`) for model-based control (e.g. computed
  torque) and for direct physical insight (mechanical design, backdrivability, stiffness).
- Obtain a **simulated** (free-run) model of `qm` from `vir` that is accurate over the
  whole trajectory, not just one-step-ahead.

## 6. Known Challenges

1. **Pure integrator** ⇒ must be identified/simulated in closed loop; open-loop
   simulation of the plant alone diverges.
2. **Two nested control loops** ⇒ cannot be reduced to the classical
   `C(s)*(qref - q)` form assumed by many system-identification toolboxes.
3. **Nonlinear, asymmetric friction** ⇒ a single symmetric `Fc*sign(qd)` term is only a
   first approximation.
4. Off-the-shelf toolboxes (CAPTAIN, CONTSID) using simple transfer functions from
   `vir` (or its derivative) to `qm`/`qdm` reach relative errors of 40-75%; only
   identifying the *whole closed loop* (`qg` -> `qm`) with a second-order model gets a
   good fit (~0.005% relative error), which the paper notes mainly reflects the quality
   of the tracking rather than of the open-loop plant model.

## 7. Reference Results (paper, IDIM-LS)

| Model                              | Relative error (train) | Relative error (cross-test) |
|-------------------------------------|:----:|:----:|
| Symmetric friction (IDIM-LS)        | 4.08% | 5.98% |
| Asymmetric friction (IDIM-LS)       | 3.11% | -     |
| Full closed-loop simulation (`DATA_EMPS`)   | position 0.014%, velocity 0.23%, acceleration 8.6%, force 5.8% | - |
| Full closed-loop simulation (`DATA_EMPS_PULSES`) | position 0.008%, velocity 0.48%, acceleration 19.8%, force 8.9% | - |

These numbers are useful sanity checks/targets: a learned black-box or grey-box
simulation model of `qm` given `vir` should aim to be competitive with (or beat) the
physically-motivated IDIM-LS baseline above, especially on the position signal, which is
the officially scored output of the benchmark.
