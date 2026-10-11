# Silverbox Benchmark

Benchmark and training code for ASIA on the Silverbox electronic nonlinear-feedback circuit.

This benchmark follows the Silverbox data set introduced by Torbjörn Wigren and Johan Schoukens in *Three free data sets for development and benchmarking in nonlinear system identification* (European Control Conference, 2013).

## Project Overview

The project lets ASIA search for a model that will simulate the output of an electronic circuit that behaves as a second-order mechanical resonator with a nonlinear (progressive) spring, `m y'' + d y' + (a + b y^2) y = u`, from the input, sampled at about 610 Hz.

- Training data: the benchmark training data. The excitation consists of 40 000 samples of filtered Gaussian noise with linearly increasing amplitude, followed by 10 random odd multisine realisations. Exact training split used: **TODO** (fill in from `prepare.py`)
- Test data: the benchmark test sets (three RMSE values are reported). Which sets: **TODO** (fill in from `prepare.py`)
- Fold structure: validation folds are built from the training records only, either as contiguous segments or by holding out one experiment at a time; number of folds: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation**; window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the simulation error (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated output on each test set, in mV
- ASIA result (extension paper): LSTM (lr 3e-3, hidden `[64, 64]`, Tanh), test RMSE [0.91, 3.78, 0.83] mV

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `Silverbox_description.md`: short benchmark description

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