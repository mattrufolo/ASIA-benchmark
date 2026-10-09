# EMPS Benchmark

Benchmark and training code for ASIA on the Electro-Mechanical Positioning System
(EMPS).

## Project Overview

The project trains a model to predict the load position `qm` from the motor force
`vir` using the official benchmark data.

- Training data: one official training trajectory (`DATA_EMPS.mat`), split once into
  a contiguous **80% train / 20% validation** split -- **no folds, no k-fold
  cross-validation**. The validation sequence spans the whole training trajectory but
  is only scored on its last 20% (see `PROGRAM.md` / `prepare.py` for details).
- Test data: one official test trajectory (`DATA_EMPS_PULSES.mat`), used only by
  `test.py`.
- Initial conditions: the previous 5 input/output samples.
- Training uses a small, fixed learning rate for a fixed (large) number of epochs --
  early stopping is intentionally not implemented, so the model has the chance to fit
  slow/small patterns over the full trajectory.

## Main Files

- [prepare.py](./prepare.py): dataset creation, cached data, single 80/20
  train/validation split, initial-condition construction
- [model.py](./model.py): black-box (LSTM/GRU/RNN/LTC), white-box (physical,
  symmetric/asymmetric friction), and grey-box (hybrid) model definitions
- [train.py](./train.py): single-run, fixed-iteration training used during the
  autoresearch loop
- [test.py](./test.py): denormalized test evaluation (mm and m) using the saved
  checkpoint
- [PROGRAM.md](./PROGRAM.md): instructions for the autonomous experimentation loop
  (at least 15 iterations: explore model families first, then tune)
- [EMPS_description.md](./EMPS_description.md): summary of the reference paper

## Reference

A. Janot, M. Gautier, M. Brunot, *"Data Set and Reference Models of EMPS"*, Nonlinear
System Identification Benchmarks, Eindhoven, 2019.

## Quickstart

```bash
uv sync
uv run prepare.py
uv run train.py
uv run test.py
```

To evaluate the best-so-far checkpoint tracked across ASIA iterations instead of the
most recent run:

```bash
uv run test.py --checkpoint-set best_so_far
```
