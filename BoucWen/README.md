# Bouc-Wen Benchmark

Benchmark and training code for ASIA on the Bouc-Wen hysteretic oscillator.

This benchmark follows the hysteretic Bouc-Wen benchmark introduced by Jean-Philippe Noël and Maarten Schoukens in *Hysteretic benchmark with a dynamic nonlinearity* (Workshop on Nonlinear System Identification Benchmarks, 2016).

## Project Overview

The project lets ASIA search for a model that will simulate the displacement `y(t)` [mm] of a single-degree-of-freedom Bouc-Wen oscillator from the external force `u(t)` [N], sampled at 750 Hz. The output carries Gaussian measurement noise; the hysteretic state `z(t)` is not measured.

- Training data: noisy estimation data simulated with the Newmark scheme of the benchmark (750 Hz). Exact files used: **TODO** (fill in from `prepare.py`)
- Test data: the two fixed, noiseless test sets of the benchmark: a random-phase multisine (5-150 Hz, 50 N RMS, 8192 samples) and a sine sweep (20-50 Hz, 40 N, zero initial conditions)
- Fold structure: validation folds are built from the training records only, either as contiguous segments or by holding out one experiment at a time; number of folds: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation** (only the input is used); window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the benchmark simulation error (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated displacement on the test set, in mm
- ASIA result (extension paper): LSTM (lr 8e-3, hidden `[32]`, ReLU), test RMSE 9.44e-5 mm. The white-box Bouc-Wen model (RK4 and explicit Euler) was too slow to converge in the one-hour budget, so the search reverted to black-box models

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `Bouc_Wen_description.md`: short benchmark description

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