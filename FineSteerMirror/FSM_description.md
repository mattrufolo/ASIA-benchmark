# CubeSpec Fine Steering Mirror (FSM) Benchmark

Summarized directly from the actual paper (Floren, Peri, De Maeyer, De
Munter, Vandepitte, Noël, *Data-driven state-space identification and
nonlinearity assessment of the CubeSpec Fine Steering Mirror*,
ISMA-USD2024, pp. 2042-2052) and its companion presentation ("Dataset
and baseline for the CubeSpec Fine Steering Mirror", Nonlinear Benchmark
Workshop 2026) -- both read directly this session, plus the official
benchmark page and the `nonlinear_benchmarks` package's own documentation.

## 1. System Description

CubeSpec's High-Precision Pointing Platform corrects residual pointing
error the spacecraft's attitude control system can't handle, using a
Fine Steering Mirror (FSM) in closed loop with a Fine Guidance Sensor.

**This benchmark**: the FSM itself, as a 3-input, 3-output MIMO system.
- **Inputs** (3): voltages applied to three piezo-actuators, connected
  via rods to the back of the mirror, enabling independent control over
  2 rotations + 1 translation. The measured voltage is taken **before**
  a 10x amplification stage (a SCADAS system generates the signal,
  amplified 10x before driving the actuators) -- "the amplifier dynamics
  are included in the estimated model" (paper's own words).
- **Outputs** (3): mirror displacement measured at three non-collocated
  points via high-accuracy capacitive probes.

**Nonlinearity source, confirmed directly from the paper's own analysis**
(not just the official page's summary): the identified best-linear-
approximation (BLA) model contains **two real (zero-frequency) poles**.
The authors explicitly attribute this to the piezo-actuators' hysteresis:
"This conjecture is substantiated by the fact that the obtained model
contains two real poles... This is consistent with the general
definition of a hysteretic system, in which the hysteresis loop persists
as the input frequency approaches zero." This directly motivates this
project's white-box model including both oscillatory AND real
(non-oscillatory) internal states -- see Section 5.

## 2. Data

**Loading mechanism**: the official `nonlinear_benchmarks.FineSteeringMirror()`
loader is not yet available for this benchmark (confirmed directly by
the user). This project instead reads the same underlying data directly
from the `fsm-benchmark-data` repo's own `data/` folder
(github.com/merijnfloren/fsm-benchmark-data) -- `.npy` files, one
`u_{level}mV_{split}.npy`/`y_{level}mV_{split}.npy` pair per amplitude
level and split. Verified directly against the actual files: shapes
match `(N=8192, 3, R, P=2)` exactly as documented, and the input RMS
values match the paper's stated 100/200/300mV levels precisely
(98.9/197.9/296.9mV measured). `fs=6400Hz` is hardcoded (confirmed from
the paper, since it isn't embedded as metadata in these files, unlike
what the official package would expose via `.sampling_time`).
`history_window=100` is a reasoned default (not an authoritative
package-provided `state_initialization_window_length` value, since
that's not available from local files) -- worth revisiting if the
official loader becomes available later.

Confirmed directly against BOTH the paper/presentation and this data's
own structure (these fully reconcile once the transient-discarding step
is accounted for -- see below):

- **Excitation**: orthogonal random-phase multisines (needed to
  uniquely solve the MIMO frequency-response-matrix identification
  problem -- see the paper's Section 2.1 for the full derivation of why
  a naive simultaneous 3-channel excitation is underdetermined).
- **`fs = 6400 Hz`**, frequencies excited up to **`fmax = 3000 Hz`**
  (DC excluded). The **most dominant resonance peaks are concentrated in
  the 750-950 Hz band** (paper's own Figure 5 / presentation slide 11).
- **3 RMS amplitude levels**: 100, 200, 300 mV (pre-amplification).
- **Raw collection**: `P=3` periods, `R=9` realizations (3 block-
  experiments x 3 repeats) per level, `N=8192` samples/period. **The
  first period is discarded to ensure steady-state** (presentation,
  slide 9) -- leaving `P=2` steady-state periods, matching the actual
  `.npy` files' own shape exactly.
- **Train/test split**: 6 realizations (2 block-experiments) for
  training, 3 (1 block-experiment) for testing -- per level, matching
  this project's own `R=6`/`R_test=3` design exactly.
- **`history_window=100`**: a reasoned default, NOT taken from an
  authoritative `state_initialization_window_length` (not available
  from local files -- see the loading-mechanism note above).

**This project's own sequence design**: the 2 steady-state periods per
realization are concatenated into one continuous 16384-sample sequence
(genuinely periodic/continuous across the boundary), giving 6 training
sequences + 3 official test sequences per amplitude level.

## 3. Fold Design

**6-fold leave-one-realization-out CV**: fold `k` holds out realization
`k` (1-6) from all 3 amplitude levels simultaneously. Tests genuine
generalization to an unseen random-phase draw, at every excitation
amplitude, every fold.

## 4. Figure of Merit and Reference Numbers

Confirmed directly against `submission_examples/FineSteeringMirror.py`:
RMSE in **micrometers**, per output channel, averaged over realizations
and periods, skipping the first `state_initialization_window_length`
samples; also NRMSE (%). This project's `test.py` matches this
convention exactly.

**Reference numbers from the paper itself** (Table 1 / presentation
slide 17, NRMSE %, "nominal" = same train/test amplitude level):

| trained on \ tested on | 100 mV | 200 mV | 300 mV |
|---|---|---|---|
| 100 mV | **7.7 / 8.2 / 9.3** | 16.6 / 26.4 / 18.5 | 26.6 / 42.0 / 29.6 |
| 200 mV | 17.1 / 26.9 / 21.3 | **7.0 / 8.4 / 8.4** | 14.5 / 21.8 / 15.6 |
| 300 mV | 26.5 / 40.8 / 31.2 | 12.1 / 18.5 / 15.0 | **4.5 / 7.0 / 5.3** |
| combined | 16.3 / 23.4 / 19.8 | 7.2 / 9.7 / 8.5 | 15.1 / 24.8 / 16.6 |

(each cell: y1/y2/y3 NRMSE %, bold = nominal/matched level)

A **nonlinear LFR model** (see Section 5) trained on all 3 levels
combined reaches **3.5-7% NRMSE across all three test levels** --
better than the linear "combined" row above, and competitive with or
better than the linear model's own nominal (matched-level) performance.
Model order for the linear baseline: `nx=28` (chosen by cross-
validation; "true order is not known").

**Important, paper-stated caveat**: the poor cross-level (off-diagonal)
generalization above is NOT necessarily true system nonlinearity -- the
authors' own hypothesis is that it's largely a **measurement-campaign
artifact** (the generator/amplifier's own behavior shifts slightly with
drive level, shifting the apparent resonance frequencies) rather than a
real nonlinear effect intrinsic to the mechanical structure. Worth
keeping in mind: this benchmark's "nonlinearity" is explicitly described
by its own authors as mild, and a substantial part of the visible
cross-level gap may not be closeable by a better nonlinear model at all.

**Honest framing from the paper's own conclusion**: "Relatively easy to
identify a linear state-space model, yet difficult to reach the noise
floor with a nonlinear model." SNRs in the data range roughly 30-50 dB.

## 5. This Project's White-Box Model

`model.py`'s `HYSTERESIS_LFR` type: 3 independent Bouc-Wen-style
hysteresis operators (one per input channel, reusing this collection's
BoucWen formulation), feeding a bank of stable discrete-time modes --
**both oscillatory (complex-pole) and real (non-oscillatory) modes**,
directly motivated by the paper's own confirmed 2-real-pole finding.
Complex-mode frequencies default to a spread centered on the confirmed
750-950Hz dominant peak band, not an arbitrary guess. Outputs are a
learnable linear combination of all modal states.

This deliberately differs in structure from the paper's own NL-LFR
"next step" model (linear core + a learned neural-network feedback
loop, `z(n)=Cz*x(n)+Dzu*u(n)`, `w(n)=f(z(n))`, `f` a small MLP) -- this
project places the nonlinearity explicitly at the input (per-actuator
hysteresis) rather than in an abstract state-feedback loop, since the
piezo-actuators are the paper's own confirmed, physically specific
source of the hysteresis. Structurally close to the same "linear
dynamics + explicit nonlinear correction" spirit, with a physically-
interpretable rather than black-box correction term.

**A genuine numerical finding from building this model** (not assumed):
the semi-implicit discretization's stability formula had to be
re-derived specifically for this project -- reusing the formula from
this collection's Silverbox/F16 white-box models (superficially similar
recursion) was verified to be unstable above ~200Hz. Re-derived
symbolically and verified numerically stable up to and beyond
`fmax=3000Hz` at the real `fs=6400Hz` -- see `model.py`'s own comments
for the corrected derivation.

## 6. References

- Floren, M., Peri, L., De Maeyer, J., De Munter, W., Vandepitte, D.,
  Noël, J.-P. *Data-driven state-space identification and nonlinearity
  assessment of the CubeSpec Fine Steering Mirror*. ISMA-USD2024,
  pp. 2042-2052.
- Floren, M. et al. *Dataset and baseline for the CubeSpec Fine Steering
  Mirror*. Nonlinear Benchmark Workshop 2026 (presentation).
- Official benchmark page: nonlinearbenchmark.org/benchmarks/fine-steering-mirror
- Dataset repo: github.com/merijnfloren/fsm-benchmark-data
- Official loader: `nonlinear_benchmarks.FineSteeringMirror()`
  (github.com/MaartenSchoukens/nonlinear_benchmarks)
- Identification/baseline toolbox: `freq-statespace`
  (github.com/merijnfloren/freq-statespace)
- Baseline models: Floren, M. (2026). *Baseline models for the Fine
  Steering Mirror benchmark dataset*. Zenodo, doi:10.5281/zenodo.19591136