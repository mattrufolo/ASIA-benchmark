# Wiener-Hammerstein Benchmark (WienerHammerBenchMark)

## 1. System Description

A real electronic nonlinear system (Schoukens, Suykens & Ljung, 2009),
built by Gerd Vandersteen, with a genuine **Wiener-Hammerstein** structure:

```
u(t) -> G1(s) [LTI] -> f[.] [static nonlinearity] -> G2(s) [LTI] -> y(t)
```

- `G1(s)`: 3rd-order Chebyshev filter (0.5 dB passband ripple, 4.4 kHz cutoff)
- `f[.]`: a diode-resistor static nonlinearity (Fig. 2 of the report) --
  memoryless, but not directly observable from `u`/`y` alone since it's
  sandwiched between the two dynamic blocks
- `G2(s)`: 3rd-order inverse Chebyshev filter (40 dB stopband attenuation
  from 5 kHz) -- has a **transmission zero** in the excited band, which the
  report explicitly flags as complicating identification (inverting it is
  hard)
- Excitation: filtered Gaussian noise, 10 kHz cutoff
- Measured at `fs = 51200 Hz` via an HPE1433A DAQ card with an internal
  anti-alias filter
- Measurement noise is very low (~70 dB below signal levels) -- the report
  notes this benchmark's focus is nonlinear-behavior identification, not
  noise rejection

## 2. Official Data (`nonlinear_benchmarks.WienerHammerBenchMark()`)

Train and test are two slices of **one continuous recording** (like CED):

- Full record: 188000 samples; the package slices off an initial ~5200-
  sample silent/transient region.
- Train: 100000 samples.
- Test: 78800 samples, `state_initialization_window_length = 50`.
- Both `u`, `y` in Volts. The official submission template reports RMSE in
  **millivolts** (`RMSE_V * 1000`).

## 3. Why the Data Is Hard

- The static nonlinearity is not directly observable (sandwiched between
  two unknown LTI blocks) -- a classic Wiener-Hammerstein identifiability
  challenge.
- `G2`'s transmission zero makes any approach that tries to *invert* the
  output dynamics (rather than *simulate* forward through them) difficult.
- High sample rate (51.2 kHz) relative to the filters' cutoff frequencies
  (4.4/5 kHz) means the system's dynamics span many samples -- filters/
  recurrent states need meaningful memory depth to capture them.

## 4. Cross-Validation Design

Train and test genuinely are two slices of one continuous recording here
(unlike EMPS/BoucWen), so -- as with CED -- contiguous folds of the
single train recording are used for leave-one-fold-out CV:

```
prepare.py:  split the 100000-sample train recording into 5 contiguous folds
train.py:    leave-one-fold-out CV across the 5 folds
test.py:     evaluate the fold-checkpoint ensemble on the official test set
```

`history_window = 50` matches the official test warmup exactly, so
`warmup_test = 0`. No `refit.py` -- same reasoning as EMPS/BoucWen:
plenty of data per fold, and ensembling is a legitimate variance-reduction
technique in its own right.

## 5. Grey-Box Model in This Project

`WienerHammersteinModel` (`model.py`) matches the system's actual
block-oriented structure directly: `G1`/`G2` are learnable causal **FIR**
filters (`FIRBlock`, a single `conv1d` -- fully vectorized, no sequential
Python loop), with a small pointwise MLP for `f[.]` in between (memoryless
by construction, matching the true diode circuit).

This design was chosen deliberately after the BoucWen project's own
experience: a sequential per-timestep recursive loop (needed there because
the ODE is genuinely nonlinear/stateful) is slow to train through: JIT
scripting only partially helped, since the *backward* pass through a deep
sequential graph is the real cost, not Python interpreter overhead. An FIR
block has no such problem -- it's a plain convolution, backpropagates in
one vectorized pass regardless of tap count, and (as a bonus) needs no
initial-condition estimation at all, since its only "memory" is its own
finite tap length, fully contained within any training window.

The trade-off: `G1`/`G2`'s true IIR responses (3rd-order Chebyshev/inverse
Chebyshev) can in principle have arbitrarily long impulse responses; an
FIR approximation needs enough taps (`n_taps_g1`/`n_taps_g2`) to capture
that adequately -- worth tuning empirically, and worth trying both smaller
(faster, cheaper) and larger (more accurate) values.

## 6. References

- J. Schoukens, J. Suykens, and L. Ljung, *Wiener-Hammerstein Benchmark*,
  15th IFAC Symposium on System Identification (SYSID), Saint-Malo,
  France, 2009.
- `nonlinear_benchmarks` package: https://github.com/MaartenSchoukens/nonlinear_benchmarks