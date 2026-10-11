# Fine Steering Mirror Benchmark

Benchmark and training code for ASIA on the CubeSpec Fine Steering Mirror (FSM).

This benchmark follows the Fine Steering Mirror benchmark introduced by Maarten Floren, Lorenzo Peri, Jonathan De Maeyer, Wouter De Munter, Dirk Vandepitte and Jean-Philippe Noël in *Data-driven state-space identification and nonlinearity assessment of the CubeSpec Fine Steering Mirror* (ISMA 2024).

## Project Overview

The project lets ASIA search for a model that will simulate the three mirror displacements measured by capacitive probes from the voltages applied to the three piezo-actuators (3 inputs, 3 outputs, MIMO), sampled at 6400 Hz.

- Training data: orthogonal random-phase multisine data at RMS levels 0.1, 0.2 and 0.3 V; per level, two of the three block-experiments are for training in the reference paper. Exact files used: **TODO** (fill in from `prepare.py`)
- Test data: the held-out FSM test data of the benchmark. Exact files used: **TODO** (fill in from `prepare.py`)
- Fold structure: several experiments and excitation levels are available, so validation folds may hold out one experiment at a time; number of folds and which experiments: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation**; window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the simulation error (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated displacements on the test data, in um (three values reported)
- ASIA result (extension paper): LSTM (lr 5e-3, hidden `[64]`, ReLU), test RMSE [0.59, 1.2, 1.8] um. This is the only benchmark where ASIA stayed worse than the baselines; the reference results come from frequency-domain subspace identification (model order 28)

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `FSM_description.md`: short benchmark description

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