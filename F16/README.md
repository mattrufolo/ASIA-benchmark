# F-16 Benchmark

Benchmark and training code for ASIA on the F-16 aircraft ground vibration test.

This benchmark follows the F-16 benchmark introduced by Jean-Philippe Noël and Maarten Schoukens in *F-16 aircraft benchmark based on ground vibration test data* (2017).

## Project Overview

The project lets ASIA search for a model that will simulate three acceleration signals measured on a full-scale F-16 (at the shaker, on the right wing and on the payload next to the nonlinear interface) from the shaker excitation, sampled at 400 Hz. The nonlinearity sits in the wing-to-payload mounting interface.

- Training data: estimation sets of the benchmark: sine-sweep and multisine records at several excitation levels (levels 3, 5, 7 are the nonlinear estimation levels in the benchmark description). Exact files used: **TODO** (fill in from `prepare.py`)
- Test data: the benchmark test sets (levels 2, 4, 6 in the benchmark description). Exact files used: **TODO** (fill in from `prepare.py`)
- Fold structure: several excitation experiments are available, so validation folds may hold out one experiment at a time; number of folds and which experiments: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation**; window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the simulation error (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated accelerations on the test sets, in m/s^2
- ASIA result (extension paper): Hybrid TCN (lr 1e-3, kernel size 5, hidden `[64]`, Tanh), test RMSE [0.09, 0.17, 0.27, 0.05, 0.11, 0.17] m/s^2. A per-timestep physical integrator was too compute-starved in the one-hour budget, so the search moved to a TCN, here with the physics model as a residual correction

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `F16_description.md`: short benchmark description

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