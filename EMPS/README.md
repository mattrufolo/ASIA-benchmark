# EMPS Benchmark

Benchmark and training code for ASIA on the Electro-Mechanical Positioning System (EMPS).

This benchmark follows the EMPS benchmark introduced by Alexandre Janot, Maxime Gautier and Mathieu Brunot in *Data set and reference models of EMPS* (2019).

## Project Overview

The project lets ASIA search for a model that will simulate the load position `qm` (stored in m, scored in mm) of a closed-loop ball-screw drive from the motor force `vir` [N], sampled at 1 kHz. The system has a pure integrator and asymmetric friction.

- Training data: the official EMPS training trajectory (`DATA_EMPS`, about 25 s, bang-bang accelerations)
- Test data: the official test trajectory (`DATA_EMPS_PULSES`, bang-bang accelerations plus applied pulses), exposed as `test` by `nonlinear_benchmarks.EMPS()`
- Fold structure: none: a single contiguous split of the training trajectory, first 80% for training windows and last 20% for validation (`validation_fraction = 0.2`)
- Prediction horizon: random training windows of at least 500 samples (`window_min_length`); validation is a free-running simulation over the whole training trajectory, scored on the held-out last 20%
- Initial conditions: the previous `history_window = 20` samples of input and output (`n_states = 40`)
- Cross-validation selection metric: validation RMSE on the held-out last 20% of the training trajectory, in normalized units
- Test metric: RMSE of the simulated position on the test trajectory, in mm (also logged in m)
- ASIA result (extension paper): CfC (lr 1e-3, hidden `[32, 32]`, Tanh), test RMSE 4.1044 mm. The physics-based white-box model underperformed because velocity and position are running sums of the estimated net force, so small residuals accumulate into a linear drift over the free-running test trajectory

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `EMPS_description.md`: short benchmark description

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