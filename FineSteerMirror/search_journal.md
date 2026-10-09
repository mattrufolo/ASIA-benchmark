# FSM Search Journal

`results.tsv` and this journal previously contained results from an
incompatible earlier benchmark (a "ParallelWH" project with a 5-fold
phase-grouped CV design and `PARALLELWH`-branch models that don't exist
in the current `model.py`). `best_kept/`, `checkpoints/`, `logs/`,
`plots/`, `cached_data/`, and stray non-FSM files in `data/` were all
leftovers from that run too. All of it has been moved to
`_stale_backup/` and cleared/reset here. Starting a fresh 15+-iteration
search on the current FSM benchmark (6-fold leave-one-realization-out
CV, `model.py`'s `FSMRecurrentModel`/`FSMHysteresisLFRModel`).

## run01 (baseline, keep)

Unmodified `model.py`/`train.py`: 1-layer LSTM, n_hidden_states=64,
hidden_sizes=[64], direct_feedthrough=True, 1800s/fold budget,
max_epochs=3000. `validation_rmse_norm_mean=0.4765` (train RMSE
matched at 0.4735, no overfitting), median best_epoch=2000, well short
of max_epochs -- early stopping (patience=15 evals) kicked in, not the
time budget. Not collapsed (collapse looks like ~1.0, the unnormalized
target variance -- seen in the old stale journal's GRU/RNN failures),
but still a fairly weak fit for this MIMO system. Test RMSE (monitoring
only): overall_mean=1.239um across the 3 levels. Baseline always kept
per PROGRAM.md; becomes the first `best_kept/` snapshot. Next: try the
white-box `HYSTERESIS_LFR` family (default config) as a meaningfully
different architecture, per the explore-then-refine plan.

## run02 (HYSTERESIS_LFR default, discard)

`n_complex_modes=3`, `n_real_modes=2`, `hidden_sizes=[32,32]`, otherwise
unmodified training loop. `validation_rmse_norm_mean=0.9954` -- looks
like the PROGRAM.md-warned collapse pattern at first glance, but the
per-fold logs show train_rmse decreasing monotonically (0.9962->0.9954
over 75 epochs) rather than freezing, and every fold only reached
epoch 75 before the 1800s/fold budget ran out. Checked the
`a_coef`/`b_coef` discretization directly: `det(M)=a_coef<1`,
`trace(M)` reduces exactly to the target `magnitude*cos(angle)` value by
construction -- genuinely stable, not the bug PROGRAM.md warns about.
Actual cause: `aggregate_metrics_across_sequences` evaluates all 18
full-length (16284-sample) sequences every `eval_every=25` epochs
through this model's sequential per-timestep Python loop, which costs
far more than one training step (4 windows, `window_min_length=1000`
default). Eval overhead was eating the time budget before enough
optimizer steps could accumulate -- a throughput problem, not an
architecture failure. Added `window_min_length_override` support to
`train.py` (`train_one_fold`, reads from `config_pars` if set, else
`general_config`'s default) and raised `eval_every` to 300 for the
retry. Next: run03, same HYSTERESIS_LFR architecture, cheaper training
loop, to get a fair read before judging this family.

## run03 (HYSTERESIS_LFR, window_min_length_override=250/eval_every=300, discard)

Worse than run02 (0.9961 vs 0.9954), despite intending to fix run02's
problem. Root-caused with a direct timing script rather than more
guessing: `window_min_length` only sets the FLOOR of the sampled window
length in `prepare.sample_training_window` (`max_window_length = hi -
start_index`, up to the full ~16000-sample remaining range) -- so
lowering it to 250 barely moved the average sampled length, and
per-step cost stayed roughly unchanged (measured directly: this model's
sequential per-timestep recursion costs ~0.0007s/timestep, so a
window averaging a few thousand samples already costs 10-20s per
training step). Separately, raising `eval_every` to 300 was actively
harmful: no fold survived long enough to reach epoch 300, so no
evaluation after epoch 1 ever ran, and every epoch of training after
that was silently discarded (the returned checkpoint is only ever the
last *evaluated* best, not the last *trained* state) -- `best_epoch`
was stuck at 1 on every fold. Fixed both problems properly this time:
added a real `window_max_length_override` to `train.py`
(`sample_fresh_training_windows`, crops from the window's own start so
`y0` -- the history-window initial condition -- stays valid) and
reverted `eval_every` to 25. Next: run04, same architecture, capped at
800-sample windows.

## run04 (HYSTERESIS_LFR, window_max_length_override=800, discard)

Throughput fix worked: ~350 epochs / 17 evals completed within the
1800s budget (vs run02's 75/4 and run03's 1/1). `train_rmse_norm`
decreases smoothly and monotonically the whole way (0.9962 -> 0.9933),
confirming this is genuinely learning, not frozen/collapsed --
`validation_rmse_norm_mean=0.9933`. But at this rate it's nowhere close
to competitive with the LSTM baseline (0.4765) inside the fixed budget.
Two live hypotheses, not yet distinguished: (a) `lr=1e-3` may be too
conservative for this modal/hysteresis parameterization, or (b) this
architecture may just need meaningfully more wall-clock than this
budget allows to reach a competitive optimum, independent of lr. Next:
try `lr=5e-3` on the same architecture as one more honest attempt before
setting HYSTERESIS_LFR aside; in parallel, cover a second black-box
family (GRU) per the explore-then-refine plan's "several families
first" guidance.

## run05 (black-box GRU, discard)

Same config as run01's LSTM but `recurrent="GRU"`. Trains cleanly, no
collapse, early-stopped at median epoch 362 -- `validation_rmse_norm_
mean=0.6674`, clearly worse than LSTM's 0.4765. Note: the pre-reset
stale journal (a different, unrelated ParallelWH benchmark project)
recorded GRU/RNN diverging entirely there; that doesn't transfer to
this benchmark's data/model.py, and indeed it didn't happen here --
worth flagging so a future run doesn't skip GRU/RNN on the strength of
that old, inapplicable finding. LSTM stays the best black-box family.
Next: `lr=5e-3` retry of HYSTERESIS_LFR (run04's config otherwise
unchanged) to settle the lr-vs-fundamentally-slow question.

## run06 (HYSTERESIS_LFR, lr=5e-3, discard)

Otherwise identical to run04. At the same median epoch (350),
`validation_rmse_norm_mean=0.9829` vs run04's 0.9933 -- only a marginal
gain from 5x the learning rate, so this isn't primarily an lr problem.
Three honest, non-buggy attempts (run02/04/06) all land in the
0.98-1.0 range, far from LSTM's 0.4765. Setting HYSTERESIS_LFR aside
for now rather than continuing to tune hyperparameters on it -- next
covering the rest of the black-box recurrent family sweep (plain RNN)
and a structurally different family (TCN/conv), then refining whichever
is strongest. Will reconsider HYSTERESIS_LFR later with a genuine
structural change (e.g. a black-box residual correction on top of the
physical core, echoing the paper's own NL-LFR "linear + learned
feedback" structure) if black-box refinement plateaus before 15
iterations.

## run07 (black-box RNN, discard)

Same config as run01 but vanilla `nn.RNN` (tanh). Trains cleanly,
best_epoch median 1112, `validation_rmse_norm_mean=0.5326` -- between
LSTM (0.4765, still best) and GRU (0.6674). Completes the black-box
recurrent sweep: LSTM > RNN > GRU here, all three genuinely trainable
on this benchmark (unlike the unrelated pre-reset project's history).
6-7 meaningfully-different-family runs done (run01 LSTM, run02/04/06
HYSTERESIS_LFR, run05 GRU, run07 RNN). Moving into the refine phase:
LSTM is the clear leader, starting with `n_hidden_states=128` (up from
64) as the first moderate hyperparameter change.

## run08 (LSTM, n_hidden_states=128, discard)

Slightly worse than run01 (0.4972 vs 0.4765) at a comparable median
best_epoch (1125 vs 2000) -- more hidden capacity isn't the bottleneck
here, if anything mildly hurts. Reverting to `n_hidden_states=64`.
Next: `crops_per_step=8` (up from 4) for smoother/less noisy gradient
estimates.

## run09 (LSTM, crops_per_step=8, discard)

Worse than run01 (0.5585 vs 0.4765). Each step costs ~2x as long with
8 windows instead of 4, so median best_epoch dropped to 1037 (from
2000) within the same 1800s budget -- fewer, smoother steps lost to
more, noisier ones here. Next: `crops_per_step=2`, the opposite
direction.

## run10 (LSTM, crops_per_step=2, discard)

Worse than run01 (0.4931 vs 0.4765), and this time hit `max_epochs=
3000` itself (still improving when cut off, not early-stopped or
time-capped) -- cheaper but noisier steps need more of them just to
reach a worse point than run01's crops=4 (which had already
early-stopped at epoch 2000, better). Confirms `crops_per_step=4` (the
original baseline) is the local optimum among {2,4,8} tested. Next:
`num_layers=2` (deeper LSTM), a different lever.

## run11 (LSTM, num_layers=2, discard)

Worse than run01 (0.5540 vs 0.4765). Same pattern as every capacity
increase so far (n_hidden_states=128 in run08, crops_per_step=8 in
run09): more per-step cost -> fewer steps within the fixed 1800s
budget (median best_epoch 912 vs run01's 2000) -> worse result. This
model is consistently step-count-limited, not capacity-limited, within
this budget. Next: `lr=2e-3` (up from 1e-3, num_layers back to 1) to
test whether faster convergence per step, rather than more capacity,
is the right lever given that observation.

## run12 (LSTM, lr=2e-3, keep)

First improvement since the baseline: `validation_rmse_norm_mean=
0.4671` vs run01's 0.4765. Median best_epoch 1800, close to run01's
2000 -- this is a better optimum reached in a similar number of steps,
not more steps, confirming the lr lever (rather than capacity) is what
this model benefits from within the fixed budget. New `best_kept/`
snapshot. Next: `lr=3e-3`, pushing the same direction once more before
checking for a plateau.

## run13 (LSTM, lr=3e-3, keep)

Further improvement: `validation_rmse_norm_mean=0.4558` vs run12's
0.4671, similar median best_epoch (1900 vs 1800). lr trend continues.
New `best_kept/` snapshot. Next: `lr=5e-3`, one more step before
checking for diminishing returns or instability.

## run14 (LSTM, lr=5e-3, keep)

Marginal gain: 0.4548 vs run13's 0.4558 (~0.2%). The lr trend is
clearly flattening. New `best_kept/` snapshot (marginally). Next:
`lr=1e-2` to check whether this is a genuine plateau or the edge before
instability, before settling the lr search and moving on.

## run15 (LSTM, lr=1e-2, discard)

Statistically identical to run14 (0.4548 vs 0.4548) -- a genuine
plateau, not just diminishing returns. Settling the LSTM lr search at
run14's `lr=5e-3` (`best_kept/`). This is the 15th iteration -- the
PROGRAM.md floor is met, but refinement has now plateaued on this
family's cheap levers (capacity/depth/crops_per_step/lr all explored:
run08/09/11 hurt, run12/13/14 helped then flattened). Per PROGRAM.md's
explicit encouragement toward major modifications before concluding,
next: a genuine hybrid model -- the `HYSTERESIS_LFR` physical core plus
a small black-box LSTM residual-correction path, zero-initialized so
training starts exactly at the physical baseline and only learns to
correct from there. Adding this as a new `model.py` class rather than
tuning further within either family alone.

## run16 (hybrid HYSTERESIS_LFR + LSTM residual, discard)

New `model.py` class `FSMHybridLFRResidualModel`: the physical core
runs unchanged, and a small (`residual_hidden_size=16`) black-box LSTM
computes a correction added directly to its output, zero-initialized so
training starts exactly at the physical prediction. Substantially
better than the pure physical core alone (0.7750 vs run06's 0.9829) --
the hybrid concept genuinely works, the correction path is learning
real structure the physical core misses. But still far behind the pure
LSTM (0.4548): median best_epoch only 325, the same throughput ceiling
as standalone HYSTERESIS_LFR, because the physical core's sequential
per-timestep Python loop still gates window size regardless of the fast
LSTM side-path. The physical core's own cost, not the hybrid idea
itself, is the limiting factor.

## Conclusion

16 iterations (exceeds the 15-iteration minimum). Covered 7
meaningfully-different-family runs (LSTM, GRU, RNN, HYSTERESIS_LFR x3,
hybrid) before refining, per the explore-then-refine plan. LSTM
refinement: capacity (`n_hidden_states`, `num_layers`) and
`crops_per_step` changes all hurt -- this model is consistently
step-count-limited within the fixed 1800s/fold budget, not
capacity-limited. Learning rate was the productive lever (1e-3 -> 2e-3
-> 3e-3 -> 5e-3 each improved; 1e-2 confirmed a genuine plateau, not
just diminishing returns). HYSTERESIS_LFR (pure and hybrid) never
became competitive: its per-timestep cost (~0.0007s, an inherent
sequential Python-loop cost) caps epoch count far below what the LSTM
gets, and no combination of lr/window-cap/hybrid-correction closed that
gap within the fixed evaluation protocol.

**Winner**: LSTM, `lr=5e-3`, `n_hidden_states=64`, `hidden_sizes=[64]`,
`direct_feedthrough=True`, `crops_per_step=4` (run14).
`validation_rmse_norm_mean=0.454753`. Final `test.py --checkpoint-set
best_so_far`: overall_mean_um=1.202 (level01=0.599, level02=1.202,
level03=1.805 -- error growing with amplitude level, consistent with
the paper's own note that cross-level degradation may partly be a
measurement-campaign artifact rather than true nonlinearity, per
PROGRAM.md). `model.py`/`train.py` left in this winning configuration;
`best_kept/` holds the matching snapshot.
