# CED Benchmark

Benchmark and training code for ASIA on the CE8 coupled electric drives.

This benchmark follows the coupled electric drives data set introduced by Torbjörn Wigren and Johan Schoukens in *Three free data sets for development and benchmarking in nonlinear system identification* (European Control Conference, 2013), with the reference models in *Coupled electric drives data set and reference models* by Torbjörn Wigren and Maarten Schoukens (Uppsala University, Technical Report 2017-024).

## Project Overview

The project lets ASIA search for a model that will simulate the pulley speed (pulse-counter output, scored in ticks/s) of two electric drives coupled by a flexible belt from the summed motor voltage, sampled at 50 Hz. The measured speed is rectified, which makes the system a Wiener system.

- Training data: short records (500 samples, 10 s each) from the five available data sets: three PRBS and two uniformly distributed inputs. Which sets are used for training: **TODO** (fill in from `prepare.py`)
- Test data: the held-out CED test data of the benchmark, never used during the search. Which sets: **TODO** (fill in from `prepare.py`)
- Fold structure: validation folds are built from the training records only, either as contiguous segments or by holding out one experiment at a time; number of folds: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation**; window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the simulation error (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated speed on the test data, in ticks/s (two values reported)
- ASIA result (extension paper): Physics-TCN (lr 1e-3, kernel size 3, hidden `[32, 32]`, Tanh), test RMSE [0.0956, 0.1003] ticks/s

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `CED_description.md`: short benchmark description

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