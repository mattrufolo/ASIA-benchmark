# Parallel Wiener-Hammerstein Benchmark

Benchmark and training code for ASIA on the parallel Wiener-Hammerstein electronic system.

This benchmark follows the parallel Wiener-Hammerstein benchmark introduced by Maarten Schoukens, Anna Marconato, Rik Pintelon, Gerd Vandersteen and Yves Rolain in *Parametric identification of parallel Wiener-Hammerstein systems* (Automatica, 2015; arXiv:1708.06543).

## Project Overview

The project lets ASIA search for a model that will simulate the output voltage `y(t)` of a two-branch parallel Wiener-Hammerstein circuit (each branch: a third-order linear filter, a diode-resistor static nonlinearity, and a second third-order linear filter) from the input voltage `u(t)`, measured at about 78 kHz.

- Training data: random-phase multisine data, flat amplitude spectrum up to 20 kHz, at 5 RMS levels linearly spaced between 100 mV and 1 V, with 20 independent realisations per level and 16 384 measured samples per period in the source paper. Exact files used: **TODO** (fill in from `prepare.py`)
- Test data: the source paper validates on a multisine realisation not used for estimation at the same 5 levels (0.1-1 V), and on a growing-envelope filtered Gaussian noise whose final RMS (1.4 V) is above the estimation range. ASIA reports five RMSE values; which test sets they are: **TODO** (fill in from `prepare.py`)
- Fold structure: validation folds are built from the training records only, either as contiguous segments or by holding out one experiment at a time; number of folds: **TODO** (fill in from `prepare.py`)
- Prediction horizon: free-run **simulation**; window length: **TODO** (fill in from `prepare.py`)
- Initial conditions: **TODO** (fill in from `prepare.py`)
- Cross-validation selection metric: average over folds of the simulation error (Eq. 1 of the ASIA paper); units: **TODO** (fill in from `prepare.py`)
- Test metric: RMSE of the simulated output on the test data, in mV (five values reported)
- ASIA result (extension paper): Grey-WH (lr 1e-3, hidden 16, Tanh), test RMSE [2.69, 4.36, 5.43, 10.07, 11.16] mV. For orientation only, the source paper reports an RMS simulation error of 0.30-1.10 mV for its proposed parallel-WH method on its multisine validation levels, and 1.34-60.12 mV for the best linear approximation; the protocols may not be identical

## Main Files of the repository

- `prepare.py`: dataset creation, windowing, initial-condition construction, cached data
- `model.py`: model definitions (black-box, white-box and hybrid structures)
- `train.py`: training and validation used during the autoresearch loop
- `test.py`: test evaluation of the retained model
- `PROGRAM.md`: instructions for the autonomous experimentation loop
- `PWH_description.md`: short benchmark description

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