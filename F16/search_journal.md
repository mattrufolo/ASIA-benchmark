# Search Journal — F16 GVT

Reset for a fresh search (prior `checkpoints/`/`logs/` were stale from an
untracked session and moved to `_stale_pre_search_backup/`; `results.tsv`
was already empty). Iteration count restarts at 0.

Note: `PROGRAM.md` in this folder describes the EMPS benchmark (verbatim),
but the actual `prepare.py`/`train.py`/`model.py`/`test.py` implement the
F16 GVT benchmark (8-fold leave-one-dataset-out CV over 4 FullMSine + 4
SineSw training trajectories, force -> 3 accelerations, mean validation
RMSE across folds as the metric). This search follows the code's actual
F16 protocol while keeping PROGRAM.md's process rules (baseline first,
keep/discard via `best_kept/`, >=15-20 iterations, explore-then-refine,
per-fold time budget from `prepare.py`'s `search_time_budget_seconds`).

## run01 (baseline, keep)

Unmodified `model.py`/`train.py`: black-box `LSTM`, `n_hidden_states=32`,
`hidden_sizes=[32]`, direct feedthrough, patience-based early stopping
(15 evals * eval_every=25 = 375-epoch patience) still active in the
unmodified trainer. Only candidate so far, kept by rule.

Result: mean validation RMSE (norm, across 8 folds) = 0.418741, std =
0.278670 (very high spread: FullMSine_train_01 0.177 vs FullMSine_train_04
0.937). Median best epoch = 575 of max_epochs=3000 -- most folds triggered
patience-based early stopping well before the epoch budget or the
30-min wall-clock fold budget, so the fixed-iteration budget is
under-used. Test RMSE (monitoring only, ensemble of the 8 fold
checkpoints on the 6 official test sets) = 0.338382 m/s^2 mean.

Kept as baseline; copied into `best_kept/`.

Next: try a genuinely different family. `model.py` already has an
unused `MODAL_WHITEBOX` physical model (2-mode modal superposition +
smooth clearance/friction nonlinearity at the mounting interface,
grounded in the benchmark's own documented mechanism) -- run that next
as run02 to see whether physics-informed structure beats the black-box
LSTM on this small, high-variance-across-datasets problem.

## run02 (MODAL_WHITEBOX, discard)

Switched `train.py`'s `config_pars["type"]` to `MODAL_WHITEBOX` (2 modes,
euler-style fixed-dt substep integration, `num_substeps=4`), everything
else unchanged from baseline.

Result: badly broken, not on modeling grounds but on compute grounds.
`F16ModalWhiteBoxModel.forward` unrolls a Python `for t in range(T)` loop
(times `num_substeps`) over the *entire* sequence. F16 train/validation
sequences are 70k-117k samples long (vs. EMPS-scale problems this model
family was presumably designed for), so a single gradient step plus one
full-sequence validation eval already consumed nearly the whole 1800s
per-fold time budget -- folds got through only 1-2 epochs before the
time-budget break kicked in and returned the best (i.e. only-evaluated)
checkpoint. Mean validation RMSE = 1.093298, much worse than baseline
0.418741. No crash, but the result is meaningless (model barely trained).
Discarded; test.py skipped (not worth spending the ensemble-eval time on
a checkpoint from ~1 optimizer step).

Conclusion: a per-timestep Python-loop physical integrator is not viable
on this benchmark's sequence lengths under the fold time budget,
regardless of the underlying physics. Any physics-based model here needs
a vectorized (loop-free) time evolution to be evaluable at all.

Next: rebuilt the *same* 2-mode linear-modal physics as a closed-form /
FFT-convolution model instead of an ODE loop -- `F16ModalConvModel`
(`MODAL_CONV` in `model.py`), added alongside (not replacing) the
existing classes. Free response (from the y0-derived initial state) uses
the analytic damped-sinusoid solution; forced response is the
convolution of the input force with each mode's analytic impulse
response, computed via `torch.fft.rfft`/`irfft` over a truncated
`kernel_length=8192` kernel (valid since a lightly-damped mode's impulse
response decays to ~0 well inside that many samples). Fully vectorized
over time, no Python loop. Nonlinear friction/clearance are dropped in
this variant (that's what decouples the modes and makes independent
per-mode convolution valid) -- a residual on top of this backbone is the
natural next step (HYBRID family) if this linear backbone shows promise.
Verified with a standalone forward/backward smoke test before launching
the full run (correct output shape, gradients flow, initial `omega`
matches the intended 5.2/7.3 Hz). Running as run03.

## run03 (MODAL_CONV, discard)

Result: fully computationally viable this time -- all folds ran the full
3000 epochs, no time-budget breaks, confirming the FFT/closed-form
rewrite fixed the compute bottleneck from run02. Mean validation RMSE =
0.710033 (train 0.849788), still clearly worse than the LSTM baseline
(0.418741). Test RMSE (monitoring) = 0.599017 m/s^2. Expected: this
variant is linear-only (no clearance/friction nonlinearity) and uses
just 2 modes, so it can't capture the benchmark's documented hard
nonlinearities or the fuller 2-15Hz mode content -- a flexible black-box
recurrent net has an easier time fitting the residual. Discarded as a
standalone model, but the backbone itself (fast, differentiable, no time
loop) is now available for a HYBRID grey-box attempt.

Next: run04, try a different black-box cell (GRU) as a fast, cheap
comparison point against the LSTM baseline before spending a run on the
more involved HYBRID (MODAL_CONV backbone + LSTM residual) idea.

## run04 (GRU, discard)

Switched `train.py`'s `config_pars["type"]` to `GRU`, otherwise identical
to the LSTM baseline. Result looked catastrophic (mean validation RMSE
1.637162, folds stopping after only 1-25 epochs) so before blaming the
architecture, benchmarked `nn.LSTM` vs `nn.GRU` vs `nn.RNN`
forward+backward directly on a 50k-step sequence in this environment:
LSTM 0.62s, GRU 16.6s (~27x slower), RNN 9.7s (~16x slower). This
PyTorch/CPU build evidently has a fast fused/optimized path for `nn.LSTM`
that `nn.GRU`/`nn.RNN` don't get here, so folds ran out of the 1800s
fold time budget after only a handful of epochs -- not evidence the GRU
architecture itself is worse, just that it's impractical to train
adequately under this benchmark's time budget on this hardware/PyTorch
build. Discarded; GRU/RNN black-box variants deprioritized going forward
for this reason (documented here so it isn't re-litigated).

Next: run05, the HYBRID grey-box idea -- added `F16HybridModel`
(`HYBRID` in `model.py`): the `MODAL_CONV` physical backbone (2 modes)
plus a small `LSTM` residual (fast per the benchmark above) that reads
`[u(t), y_phys(t)]` and predicts a correction for the physics
backbone's un-modeled nonlinear clearance/friction and un-modeled higher
modes. This is the "major modification: mixing physical and black-box
components" PROGRAM.md explicitly calls out, and is a natural next step
after run03 showed the linear physics backbone is fast but underfits
alone.

## run05 (HYBRID, keep -- new best)

`F16HybridModel`: `MODAL_CONV` backbone (2 modes, same as run03) + a
32-unit `nn.LSTM` residual reading `[u(t), y_phys(t)]`, residual output
layer zero-initialized so training starts from the pure physics
prediction. Verified with a standalone forward/backward smoke test
before launching.

Result: mean validation RMSE = 0.359412 (std 0.246890, train 0.365812),
beating the LSTM baseline's 0.418741 -- new best. Test RMSE (monitoring)
= 0.265894 m/s^2, also better than baseline's 0.338382. Per-fold pattern
mirrors both run01 and run03 (FullMSine_train_04 is the hardest fold in
every family so far, ~0.82-0.94 RMSE; SineSw_train_01 the easiest,
~0.04-0.07), suggesting the difficulty spread is dataset-driven rather
than architecture-driven. Kept; copied into `best_kept/`.

Confirms the grey-box hypothesis: the physics backbone alone (run03,
0.710) underfits, but combined with a residual that only has to learn
the *correction* (nonlinear friction/clearance + higher modes) rather
than the whole input-output map from scratch, it beats a same-size pure
black-box LSTM. This is now the most promising family -- per the
explore/refine strategy, worth a few refinement runs (larger residual,
more modes, asymmetric-ish nonlinearity in the backbone) before moving
on to a different family again.

Next: run06, refine the HYBRID -- try `n_modes=4` in the backbone (the
paper's own excited band is ~2-15Hz, wider than the 2 named modes) to
see whether extra physics capacity helps beyond what the LSTM residual
alone can absorb.

## run06 (HYBRID, n_modes=4, keep -- new best)

Backbone modes 2 -> 4 (extra 2 modes spread 8-14Hz per the default
`omega_init_hz` spacing in `model.py`), everything else unchanged from
run05. Result: mean validation RMSE 0.348700 (vs 0.359412), test
(monitoring) 0.255904 m/s^2 -- both improved. Kept, copied into
`best_kept/`.

## run07 (HYBRID, residual_hidden_size=64, discard)

Refinement attempt: residual LSTM hidden size 32 -> 64 (more residual
capacity), backbone unchanged (`n_modes=4`). Result: worse (0.367246 vs
0.348700), and `best_epoch` dropped to ~250 (vs ~437 for run06),
suggesting the larger residual starts overfitting/destabilizing sooner
rather than needing more capacity. Discarded; reverted to `best_kept`
(run06's `n_modes=4`, `residual_hidden_size=32`).

Two refinement attempts on HYBRID (n_modes, residual size) are in:
n_modes helped, residual size didn't. Per the explore/refine strategy,
before spending more runs narrowly tuning HYBRID, next try a genuinely
different family for breadth (dilated causal CNN -- fully vectorized
convolutional black-box, no recurrence, a different inductive bias than
everything tried so far) as run08, then return to HYBRID refinement
(e.g. n_modes=6, or asymmetric/extra nonlinear terms in the backbone) if
the CNN doesn't beat 0.348700.

## run08 (TCN, keep -- new best by a wide margin)

Added `F16CausalCNNModel` (`TCN` in `model.py`): a WaveNet-style stack of
dilated causal conv1d residual blocks (`channels=32`, `n_layers=9`,
`kernel_size=3` -> receptive field 1023 samples = ~2.6s), y0-conditioned
via an additive FiLM-style bias projected into every block, no
recurrence anywhere (fully vectorized -- a single sequence of `conv1d`
calls over the whole window, no `for t in range(T)` and no sequential
RNN cell). Benchmarked standalone first: ~0.6-1.3s forward+backward for
T=50k-100k, on par with `nn.LSTM`, confirming it's not going to hit the
per-fold time budget wall the way `MODAL_WHITEBOX`/GRU/RNN did.

Result: mean validation RMSE 0.248394 (std 0.151161, train 0.224127) --
by far the best family so far, well ahead of HYBRID's 0.348700 and
almost half the original LSTM baseline (0.418741). Test RMSE
(monitoring) = 0.171030 m/s^2, also far ahead of every prior run. Kept,
copied into `best_kept/`.

This is a big enough jump that it changes the exploration plan: dilated
convolution (parallel receptive field over the crop, no vanishing/
exploding-gradient-prone recurrence, no state to carry across a crop
boundary) looks much better suited to this benchmark's short random
training crops than any recurrent or physics-integrator approach tried
so far. Next few runs: refine TCN (receptive field size via more layers
or larger kernel, channel width, activation) before trying anything
else, per the explore/refine rule -- a promising family found, so spend
several runs tuning it.

Next: run09, refine TCN -- try widening (`channels` 32 -> 64) since a
convolutional stack is comparatively cheap and receptive-field coverage
already looked reasonable (1023 samples).

## run09 (TCN, channels=64, keep)

`channels` 32 -> 64. Small improvement: val RMSE 0.246480 (vs
0.248394), test 0.164408 m/s^2. New best, kept.

## run10 (TCN, n_layers=11, running)

Widening gave only a small gain, so next trying more receptive field
instead: `n_layers` 9 -> 11 (receptive field 1023 -> 4095 samples =
~10.2s at 400Hz), `channels` stays 64. Motivation: lightly-damped
structural modes (paper's own ~1% damping estimate used as the
`MODAL_CONV`/`MODAL_WHITEBOX` init) ring down over several seconds, and
1023 samples (~2.6s) may be cutting that off early.

Result: essentially flat (0.246210 vs 0.246480 for run09), well inside
noise. Not a meaningful improvement -- per the strategy, stop refining
pure depth/width of the TCN. Discarded, reverted to `best_kept` (run09:
`channels=64`, `n_layers=9`).

## run11 (HYBRID_TCN, running)

A different modification instead of another size knob: added
`F16HybridTCNModel` (`HYBRID_TCN`) -- same `MODAL_CONV` physics backbone
as `HYBRID` (run05/run06), but the residual is now a dilated causal
conv1d stack (reusing `CausalConvBlock`) reading `[u(t), y_phys(t)]`
instead of an LSTM. Motivation: the standalone TCN (run08/run09) clearly
beat the standalone LSTM, so testing whether a TCN-based residual on top
of the physics backbone can beat the best pure TCN (0.246480) outright,
combining both advantages (physics inductive bias + the architecture
that's proven best at fitting the residual). Verified with a standalone
forward/backward smoke test before launching.

Result: mean validation RMSE 0.243444 (std 0.165384, train 0.228953),
test (monitoring) 0.162275 m/s^2 -- a small but real improvement over
the pure TCN (0.246480). Kept, copied into `best_kept/`. Smaller gain
than run08's jump, but the physics backbone is essentially free
(FFT-based, no time-loop cost), so no real downside to keeping it.

Noticed something worth acting on: `best_epoch_median` has been well
under `max_epochs=3000` for both TCN-family runs so far (run09: 337,
run11: 237) -- much earlier than the LSTM baseline, which was actually
time-budget-limited (stopped at ~575-625, matching its *last* epoch
too, i.e. still improving when the 1800s ran out). `early_stopping_patience=15`
(inherited unchanged from the original file) is a patience-based early
stop, and PROGRAM.md is explicit that this benchmark's `train.py` should
NOT stop early on a validation-performance basis ("Fixed-iteration
training (no early stopping)" -- keep `lr` small and let a long fixed
run pick up slow/small patterns). Since TCN/HYBRID_TCN are cheap per
step (~0.5-1.3s forward+backward vs. the 1800s budget), disabling
patience should let them use dramatically more of the fold time budget.

Next: run12, same best config (HYBRID_TCN) but with
`early_stopping_patience` effectively disabled (set very large) so
training runs until `max_epochs` or the fold time budget, matching
PROGRAM.md's intended protocol -- checking whether the extra training
meaningfully improves on 0.243444.

Result: essentially flat (0.244955 vs 0.243444, within noise). Checked
*why*: folds only reached epoch ~225-375 total before the 1800s fold
budget stopped them (not patience, which was disabled) -- so this model
family was already time-budget-limited, not patience-limited; run11's
patience=15 almost never actually got to fire before the clock ran out
anyway. Discarded (no meaningful change), reverted to `best_kept`
(run11, patience restored to 15 since it's inert here regardless).

Since wall-clock budget (not epoch count) is the real constraint for
this family, next try reducing *evaluation* overhead instead of
disabling patience: each eval (`eval_every=25`) runs a full forward
pass over all 7 training sequences plus the validation sequence
(~70k-117k samples each) purely to log/checkpoint -- at `eval_every=25`
that's a lot of the budget spent on bookkeeping rather than gradient
steps. Try `eval_every=50` (via `train.py`'s already-supported override)
as run13, freeing more of the 1800s for actual training steps.

Result: essentially flat again (0.243972 vs 0.243444, within noise).
Discarded, reverted to `best_kept` (run11). Two efficiency-oriented
attempts (disabling patience, cutting eval overhead) both landed within
noise of run11 -- this model family appears to be sitting near a
plateau for the current architecture at this time budget, not starved
for extra steps.

Next: run14, back to an architecture-level change on the current best --
`n_modes` 4 -> 6 in the `HYBRID_TCN` backbone (this combination, more
physics modes + TCN residual, hasn't been tried; only tested
mode-count with the LSTM residual before, in run06).

Result: worse (0.249704 vs 0.243444). Once a capable TCN residual is
already covering the gap, adding more physics modes doesn't help --
consistent with the residual already absorbing whatever the extra modes
would represent, plus more backbone parameters competing for the same
optimization budget. Discarded, reverted to `best_kept` (run11,
`n_modes=4`).

Next: run15, a different lever on the same config -- widen the TCN
residual's per-layer receptive field via `kernel_size` 3 -> 5 (more
context per layer, vs. run10's failed attempt to get there through more
layers instead).

## run15 (HYBRID_TCN, kernel_size=5, keep -- new best)

Result: mean validation RMSE 0.226512 (std 0.146989, train 0.211456),
test (monitoring) 0.145988 m/s^2 -- a clear, non-noise-level improvement
over run11 (0.243444), the biggest gain since run08/run11. Kept, copied
into `best_kept/`.

Notable: run10 tried to get a bigger receptive field via more *layers*
(9->11, 1023->4095 samples) on the plain TCN and got nothing (flat).
This run gets there via a wider *kernel* instead (3->5, receptive field
1023->2045 samples on the `HYBRID_TCN` residual) and gets a real gain.
Widening the kernel gives every layer more local context per dilation
step rather than just pushing the same 3-tap view further out --
apparently a materially different (and better) way to grow context here
than adding depth. Reached at least 15 completed iterations (the
required floor) with this run; continuing a bit further since this
kernel-size lever looks promising and hasn't been pushed further yet.

Next: run16, push `kernel_size` further (5 -> 7) to see whether the
trend continues or plateaus.

## run16 (HYBRID_TCN, kernel_size=7, discard)

Result: worse (0.231135 vs 0.226512) -- `kernel_size=5` is a local
optimum on this axis; widening further doesn't help (more parameters
per layer, likely trading off against effective step count in the fixed
budget: `best_epoch_median` dropped to 212 vs 250). Discarded, reverted
to `best_kept` (run15, `kernel_size=5`).

Now past the 15-run floor with a clear, consistently-refined winner
(`HYBRID_TCN`, `n_modes=4`, `channels=64`, `n_layers=9`,
`kernel_size=5`). One more refinement worth trying given the remaining
budget: `channels` 64 -> 96 combined with the now-best `kernel_size=5`
(the earlier channel-width test, run09, was done before kernel_size was
tuned, so this combination hasn't been checked).

## run17 (HYBRID_TCN, channels=96, discard)

Result: worse (0.250417 vs 0.226512), `best_epoch_median` dropped to
~150 -- wider channels overfits/destabilizes faster within the fixed
time budget, mirroring run07's residual-widening failure in the
LSTM-residual HYBRID. Discarded, reverted to `best_kept` (run15).

## Search concluded (17 iterations, above the 15-20 floor)

Final winner: `HYBRID_TCN` -- `MODAL_CONV` physics backbone (4 modes,
closed-form/FFT, no python time loop) + dilated causal conv1d residual
(`channels=64`, `n_layers=9`, `kernel_size=5`, receptive field 2045
samples), y0-conditioned via additive FiLM-style bias.

- Validation RMSE (norm, mean over 8 folds): **0.226512** (run15), vs.
  0.418741 for the unmodified LSTM baseline (run01) -- a 46% reduction.
- Official test RMSE (`test.py --checkpoint-set best_so_far`, monitoring
  only, m/s^2 per official dataset order): `[0.0999, 0.1766, 0.2710,
  0.0503, 0.1111, 0.1670]`, overall mean 0.145988 m/s^2 (vs. 0.338382
  for the baseline).

Summary of what was learned across the 17 runs:
- A per-timestep Python-loop physics integrator (`MODAL_WHITEBOX`) is
  computationally infeasible on F16's 70k-117k-sample sequences under
  the fold time budget (run02) -- any physics component needs a
  vectorized/closed-form time evolution to be usable at all here. The
  `MODAL_CONV` rewrite (closed-form free response + FFT convolution,
  same underlying 2-mode physics) fixed that (run03).
- This PyTorch/CPU build's `nn.GRU`/`nn.RNN` are ~15-27x slower than
  `nn.LSTM` for long sequences (run04) -- an environment quirk, not a
  modeling finding, but one that ruled out GRU/RNN as practical
  black-box options here.
- Pure physics (linear-only, no friction/clearance nonlinearity)
  underfits alone (run03: 0.710), but as a backbone under a residual
  corrector it's a net win at essentially no extra compute cost (run05,
  run06, run11, run15 all beat their pure-black-box counterparts by the
  same margin or more).
- Dilated causal convolution (`TCN`) was the single biggest jump in the
  whole search (run08: 0.248 vs. HYBRID's 0.349) -- it suits this
  benchmark's short random-crop training much better than any recurrent
  or ODE-integrator approach tried.
- Refining a promising family doesn't always mean "bigger": more
  physics modes, more residual channels, more TCN layers, and a wider
  TCN kernel (7) all made things worse or flat once the main structural
  choice (TCN residual, kernel_size=5) was right; only `kernel_size`
  3->5 and backbone `n_modes` 2->4 (within the LSTM-residual variant)
  produced real gains beyond the initial family switches.

All results are in `results.tsv`; `best_kept/model.py` and
`best_kept/train.py` hold the winning configuration; `checkpoints/best_so_far/`
holds its 8 fold checkpoints.

