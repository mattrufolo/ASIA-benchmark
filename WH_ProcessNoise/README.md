# Wiener-Hammerstein with Process Noise Benchmark

Benchmark and training code for ASIA on the Wiener-Hammerstein circuit with dominant process noise.

This benchmark follows the benchmark introduced by Maarten Schoukens and Jean-Philippe Noël in *Wiener-Hammerstein benchmark with process noise* (Workshop on Nonlinear System Identification Benchmarks, 2016).

## Project Overview

The project lets ASIA search for a model that will simulate the output voltage `y(t)` of a Wiener-Hammerstein circuit, in which filtered Gaussian noise enters before the static nonlinearity, from the input voltage `u(t)`, sampled at 78 125 Hz.

- Training data: estimation data with process noise. Exact files used: **TODO** (fill in from `prepare.py`)
- Test data: two noiseless test sets measured without process noise: a random-phase multisine and a sine sweep (0.71 V RMS, DC-15 kHz)
- Fold structure: validation folds are built from the training records only, either as contiguous segments or by holding out one experiment at a time; number of folds: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation**; window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the simulation error (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated output on each test set, in mV
- ASIA result (extension paper): Grey-WH (lr 1e-3, hidden `[32, 32]`, Tanh), test RMSE [0.016, 0.034] mV, one value per test set. Published results under the same protocol: STORN [0.034, 0.051], PNLSS [0.022, 0.038], VAE-RNN [0.050, 0.059]

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `WH_PN_description.md`: short benchmark description

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