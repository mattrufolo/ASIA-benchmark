# Wiener-Hammerstein Benchmark

Benchmark and training code for ASIA on the Wiener-Hammerstein electronic circuit.

This benchmark follows the Wiener-Hammerstein benchmark introduced by Johan Schoukens, Johan Suykens and Lennart Ljung in *Wiener-Hammerstein benchmark* (15th IFAC Symposium on System Identification, 2009).

## Project Overview

The project lets ASIA search for a model that will simulate the output voltage `y(t)` of a circuit made of a linear filter, a diode-based static nonlinearity and a second linear filter, from the input voltage `u(t)`, sampled at 51 200 Hz.

- Training data: estimation data: samples 1-100 000 of the 188 000-sample record (measurement noise about 70 dB below the signal)
- Test data: samples 100 001-188 000 of the record, never used during the search
- Fold structure: the estimation record is a single experiment, so validation folds are contiguous segments of it; number of folds: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation** (no past measured outputs); window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the simulation RMSE (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated output on the test data, in mV (start-up transients are excluded: the sum starts at sample 101 001)
- ASIA result (extension paper): Grey-WH + GRU residual (lr 3e-3, hidden 16, Tanh), test RMSE 2.81 mV. Grey-WH is a static MLP between two FIR filters implemented as 1-D causal convolutions

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `WH_description.md`: short benchmark description

## Useful Branches for Reproducibility

The repository includes two additional branches that are useful for reproducing or inspecting the ASIA workflow:

- `best`: best-performing model branch
- `final`: final branch after ASIA completed the experimentation cycle

To switch to the best branch and reproduce the learning of the best model:

```bash
git switch best
uv sync
uv run train.py
```

To switch to the final branch and inspect the final outputs produced by the AI agent after the end of the experiments:

```bash
git switch final
```

To return to the main branch:

```bash
git switch main
```