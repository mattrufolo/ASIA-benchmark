# Wiener-Hammerstein Benchmark With Process Noise

## 1. System Description

A real electronic Wiener-Hammerstein circuit (Schoukens & Noel,
2016/2017) -- structurally the same block-oriented family as this
collection's plain `WienerHammerBenchMark` project, but with a
fundamentally different noise structure, confirmed directly from the
paper's own Fig. 2 and Section 4.2:

```
u --> R(s) --> [+ process noise e_x(t)] --> f(x) --> S(s) --> y
                                              ^
                                    diode-resistor static
                                    nonlinearity (not directly
                                    accessible from either the
                                    measured input or output)
```

- `R(s)`, `S(s)`: front/back LTI blocks (both 3rd order in the real
  circuit). `S(s)` has a transmission zero -- the output dynamics are
  hard to invert, the same challenge the plain WienerHammerBenchMark's
  back filter has.
- **`e_x(t)`: the DOMINANT noise source, entering INSIDE the system,
  BEFORE the static nonlinearity.** This is the defining difference from
  the plain WienerHammerBenchMark project (which only has small
  measurement noise). Because the disturbed signal is then passed through
  a nonlinearity, this is not equivalent to simple additive output noise
  -- the noise interacts with the system's own nonlinear dynamics.
- Two much smaller, separate measurement-noise sources `e_u(t)`, `e_y(t)`
  on the input/output channels -- "neglectable" per the paper.
- Followed by a zero-order hold + analog reconstruction filter, cutoff
  20kHz.
- `fs = 78125 Hz` (confirmed directly from the official `.mat` files).

## 2. Official Data

The official zip (`data.4tu.nl`, article 12952124) contains several
measurement-campaign files; confirmed directly by loading the actual
`.mat` structs (each has fields `r, u, y, fs` -- `r` is the reference/
programmed input before measurement noise, `u`/`y` are the measured
signals actually used here):

| File | Content | Shape | Noise |
|---|---|---|---|
| `WH_EstimationExample.mat` | 10 realizations, 2 periods x 4096 pts | (8192, 10) | process noise present |
| `WH_MultisineFadeOut.mat` | 2 realizations, fading to zero at the end | (8192, 2) | process noise present |
| `WH_TestDataset.mat` | 2 columns = [multisine, sine-sweep] | (16384, 2) | **noiseless** |

Per the official `README.txt`: *"Noiseless test dataset (both multisine
and swept sine inputs, not to be used during the model estimation)."*

Several additional measurement-campaign files also exist in the zip
(`WH_CombinedZeroMultisineSinesweep.mat`: 5 x 57346 pts; `WH_SineInput_
meas.mat`: 20 frequencies x 5 amplitudes, 65536 pts; `WH_SineSweepInput_
meas.mat`: 100 periods x 64161 pts; `WH_Triangle_meas.mat` /
`WH_Triangle2_meas.mat`: multisine superposed on a triangle input;
`WH_ZeroMeas.mat`: 2 x 4096 pts, zero input) -- not used by this
project's default protocol (see Section 4), but cached and available for
experimentation.

## 3. Why the Data Is Hard

- **Process noise inside the nonlinearity's input path**: a model that
  only ever sees noisy training data cannot, even in principle, perfectly
  reproduce any single training realization -- the best it can do is
  learn the underlying deterministic `R -> f -> S` mapping, averaged
  across noise realizations. This is a genuinely different objective than
  ordinary output-noise-only identification.
- **The static nonlinearity has no direct probe**: unlike some benchmarks
  where the intermediate signal might be inferable, `f(x)`'s input/output
  are never directly measured.
- **Transmission zero in `S(s)`**: makes exact inversion of the output
  dynamics impossible, the same identifiability challenge as the plain
  WienerHammerBenchMark project.
- **Test data is deliberately noise-free**: the benchmark's whole point is
  to check whether a model trained under noisy conditions correctly
  identifies the underlying deterministic system, not whether it
  memorizes noise patterns from training.

## 4. Cross-Validation Design (Fold Choice, Explained)

**Chosen default: `WH_EstimationExample.mat`'s 10 realizations, grouped
into 5 folds of 2 realizations each, leave-one-fold-out CV.**

Two candidate estimation datasets exist in the official zip:
`WH_MultisineFadeOut.mat` (2 realizations) is what most third-party
papers citing this benchmark actually used, likely just because it was
the first/most convenient file found -- but `WH_EstimationExample.mat`
(10 realizations) is **literally named for this exact purpose** and gives
far richer folding options. With only 2 realizations, "folding" is barely
possible at all (2-way leave-one-out at most); with 10, genuine 5-fold CV
becomes possible while still holding out real, independent realizations
each time.

This directly answers the "is folding better, or something similar"
question this project was asked to investigate: **it depends on which
estimation set is used**, and that itself is worth testing explicitly --
see `program.md`'s Search Strategy for how this is set up as one of the
required early experiments (comparing the 10-realization 5-fold design
against the 2-realization literature-standard set, and against simpler
alternatives like a single held-out realization).

No `refit.py`: with 10 realizations available and 4/5 always used for
training, holding one fold back is not a meaningful data sacrifice.

## 5. Grey-Box Model in This Project

`WienerHammersteinProcessNoiseModel` (`model.py`) mirrors the true
`R -> f -> S` structure directly: each LTI block is a learnable causal
**FIR** filter (fully vectorized `conv1d`, no sequential Python loop, no
initial-condition estimation needed at all), the same design already
proven on the plain WienerHammerBenchMark project, for the same reasons
(BoucWen/Silverbox/EMPS found that a sequential per-timestep recursion
for a genuinely stateful/nonlinear system is both fragile and slow to
backpropagate through -- an FIR approximation of the LTI blocks
sidesteps both problems).

**This model is purely deterministic** (no explicit process-noise term).
It can only learn the average/expected `u -> y` mapping across the noise
realizations it trains on -- appropriate, since the noiseless test data
is specifically designed to check exactly that. `n_taps_r`/`n_taps_s`
trade FIR approximation quality for the true LTI responses against model
size/training cost.

**A genuinely different avenue worth exploring** (not built in by
default, per this collection's "let the search decide" convention): a
model that explicitly represents process noise as a per-timestep learned
perturbation at the nonlinearity's input, or a stochastic state-space
formulation -- since the benchmark's core challenge is specifically about
this noise structure, an architecture that models it explicitly, rather
than only implicitly averaging it away via multiple training
realizations, could plausibly do better. Worth an explicit experiment.

## 6. References

- M. Schoukens and J.P. Noel, *Three Benchmarks Addressing Open
  Challenges in Nonlinear System Identification*, IFAC-PapersOnLine
  50-1, pp. 446-451, 2017 (20th IFAC World Congress) -- read directly for
  this project; confirms the `R -> [+e_x] -> f -> S` structure (Fig. 2)
  and the three identification challenges (Section 4.2).
- M. Schoukens, *Wiener-Hammerstein Benchmark With Process Noise*, 2016.
- Dataset: https://data.4tu.nl/articles/_/12952124 (official
  `WienerHammerBenchmark.pdf` and README included in the zip). This
  project reads the extracted `.mat` files DIRECTLY (see `prepare.py`'s
  `RAW_MAT_DIR`) -- not through the `nonlinear_benchmarks` package's own
  `WienerHammerstein_Process_Noise()` loader, since this benchmark isn't
  reliably available through that package.