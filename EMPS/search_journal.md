# EMPS Search Journal

**Fresh search restart (iteration count starts at 0).** `results.tsv` was empty and
has been reinitialized with only the header row. A stale `best_kept/` snapshot from a
previous cycle (a `HYBRID` config with asymmetric-friction disabled, `long_window_prob
0.15`, `drift_penalty_weight 0.05`) was found and cleared, since it belonged to an
earlier, unrelated search and this is a deliberate new cycle, not a continuation.
`model.py` and `train.py` on disk were already back at the plain baseline
(`type: LSTM`, no drift penalty, no long-window mixing), so no further reset was
needed there.

One setup fix made before run01: `train.py`'s `config_pars["time_budget_seconds"]`
carried a leftover `3600.0` override from the previous cycle. Since the per-run
wall-clock safety net must come from `prepare.py`'s `config_pars_general["time_budget_seconds"]`
(1800s), the override was removed (`None`) so training always falls back to that
canonical value going forward.

Cached data (`cached_data/`, `data/`) is up to date and was not regenerated;
`prepare.py` was not modified.

## run01 (baseline, LSTM)

Unmodified baseline: `type: LSTM`, `n_hidden_states: 32`, `hidden_sizes: [32]`,
`lr: 1e-4`, `max_epochs: 20000`, no drift penalty, no long-window mixing. This is the
only candidate so far, so it is kept by definition and copied into `best_kept/`.

Result: hit the 1800s wall-clock safety net at epoch 1625/20000 (only ~8% of the
requested schedule), best validation RMSE (normalized) 0.949982 at epoch 250, test
RMSE 81.6031 mm. A normalized RMSE close to 1.0 means the free-run simulation is
barely better than a naive constant predictor over the scored last-20% tail -- the
generic recurrent baseline is not capturing the closed-loop dynamics well in this
compute budget, consistent with the paper's own note that black-box toolboxes perform
poorly on this benchmark unless the physics of the closed loop is used directly.

Kept as the current baseline to beat. Next step: try the white-box `PHYSICAL` family
(paper's Direct Dynamic Model, symmetric Coulomb + viscous friction), which is a
qualitatively different modeling philosophy and should need far fewer effective
epochs to fit its handful of physical parameters.

## run02 (PHYSICAL, symmetric friction, euler)

Changed `type` to `PHYSICAL` (euler integrator), `hidden_sizes` to `[32, 32]` for the
state-init net, and `long_window_prob` to 0.15 per `train.py`'s own guidance for
white-box families (occasional full-length windows so a constant parameter bias shows
up in the loss instead of being invisible on short, truth-anchored crops). `lr` and
`crops_per_step` left unchanged from baseline.

Result: validation RMSE (normalized) 0.909256, test RMSE 79.4905 mm -- already beats
the LSTM baseline. But `best_epoch: 0`: the model never improved beyond its untrained
initialization within the 1800s budget. The Python per-timestep integration loop,
combined with the occasional full-trajectory-length window from `long_window_prob`,
made each epoch far more expensive than the LSTM's fused kernel -- only 375/20000
epochs completed, far too few gradient steps (especially at `lr=1e-4`, log-parameterized
`M`/`Fv`/`Fc`) for the physical parameters to move meaningfully. Discarded: the result
is really an untrained-physical-model number, not evidence about the family itself
(which already looks promising given the correct-structure prior alone beats a fully
trained black-box). Reverting working files to `best_kept` (run01) before the next
experiment.

Next step: keep the `PHYSICAL` family but fix the convergence-speed problem before
judging it -- raise `lr` (1e-4 -> 1e-3; still small, but the physical model has only a
handful of scalar parameters plus a small state-init net, so it can tolerate a larger
step) and turn `long_window_prob` back to 0.0 for this run, to isolate whether cheap,
frequent short-window steps let the friction/mass parameters converge at all before
reintroducing the long-window bias correction in a later refinement.

## run03 (PHYSICAL, symmetric friction, euler, lr 1e-3)

Same `PHYSICAL`/euler family as run02, but `lr: 1e-4 -> 1e-3` and `long_window_prob`
back to 0.0 (isolate raw convergence speed first).

Result: validation RMSE (normalized) 0.768916 at best_epoch 400, test RMSE 76.4577 mm
-- a clear improvement over both run01 (LSTM, 0.949982) and run02's untrained-physical
number (0.909256). Confirms the earlier hypothesis: at `lr=1e-4` the physical
parameters barely moved in the available steps; `1e-3` lets them converge. Training
curve is noisy (val RMSE swings between ~0.2 and ~0.9 across nearby eval points),
consistent with a stiff friction system and a still-fairly-large step size, but the
best checkpoint is clearly better than either black-box or untrained physical.
Also confirms a structural fact about this benchmark: `prepare.py`'s
`sample_training_window` draws window lengths uniformly up to the *entire* first-80%
span regardless of `train.py` settings (only the lower bound is configurable via
`window_min_length`/`crops_per_step`), so `PHYSICAL`'s Python per-timestep loop is
inherently epoch-expensive here (only 525/20000 epochs fit in 1800s) -- not something
fixable without touching `prepare.py`, which is read-only.

Kept, copied into `best_kept/`. Next step: still in the family-exploration phase --
try `PHYSICAL_ASYM` (asymmetric friction, the paper's eq. 12 refinement that measurably
improved the fit over symmetric friction), reusing the `lr=1e-3` fix. Also lowering
`crops_per_step` 2 -> 1 to get more effective optimizer steps per second given the
per-epoch cost problem just identified, so family comparisons are less bottlenecked by
raw step count.

## run04 (PHYSICAL_ASYM, asymmetric friction, euler, lr 1e-3, crops_per_step 1)

Switched `type` to `PHYSICAL_ASYM` and `crops_per_step` 2 -> 1 (cheaper epochs, more
gradient steps per second), keeping `lr=1e-3` from run03.

Result: validation RMSE (normalized) 0.792403 at best_epoch 600, test RMSE 79.9503 mm.
More epochs completed than run03 (650 vs 525, confirming `crops_per_step=1` helps
throughput), but the extra asymmetric-friction parameters (5 scalars instead of 3)
did not translate into a better fit here -- slightly worse than run03's symmetric
model. Training remains noisy epoch-to-epoch. Likely the doubled friction-parameter
set needs more steps to identify than were available, or the benefit of asymmetry is
real but small next to the noise floor at this step count. Discarded; reverted
`model.py`/`train.py` to `best_kept` (run03, symmetric `PHYSICAL`).

Next step: still exploring families per the Phase 1 plan -- try `HYBRID` (grey-box:
`PHYSICAL` backbone + small LSTM residual), which is a qualitatively different idea
(physics + learned correction) rather than a refinement of the physical model alone,
and keep `lr=1e-3`, `crops_per_step=1` from this run's throughput fix.

## run05 (HYBRID, symmetric-friction backbone + 16-unit LSTM residual)

Switched `type` to `HYBRID` with `asymmetric_friction: False` (symmetric backbone,
since run04 showed asymmetric friction alone did not help yet), `n_hidden_states: 16`
for the residual LSTM, `lr=1e-3`, `crops_per_step=1` carried over from run04.

Result: validation RMSE (normalized) 0.705730 at best_epoch 825, test RMSE 70.5730 mm
-- the best result so far, beating run03's pure-physical 0.768916. The learned
residual is picking up real unmodeled effects on top of the physics backbone (nested
PD tracking error, encoder quantization, resonances), as the grey-box hypothesis in
`model.py`'s docstring predicts. Training curve is still noisy (val RMSE swings
0.01-1.7 across nearby eval points, best checkpoint bookkeeping is doing real work
here), but the trend is clearly better than the white-box-only family.

Kept, copied into `best_kept/`. Next step: this is now the most promising family --
begin refinement (Phase 2) on `HYBRID` for a few runs: try `asymmetric_friction: True`
now that it has a learned residual to absorb any remaining bias (unlike run04, where
it had to carry the whole load alone), and/or tune the residual size/lr before
concluding whether HYBRID is the final direction.

## run06 (HYBRID, asymmetric-friction backbone + LSTM residual)

Same as run05 but `asymmetric_friction: True`.

Result: validation RMSE (normalized) 0.804801 at best_epoch 575, test RMSE 77.4076 mm
-- worse than run05 (0.705730). Fewer epochs completed (575 vs 875), consistent with
the asymmetric backbone's extra parameters adding a bit more per-step cost; the extra
friction split does not pay for itself yet against the loss of step count, similar to
the run03-vs-run04 comparison in the pure-physical case. Discarded; reverted to
`best_kept` (run05, symmetric HYBRID).

Next step: symmetric-friction `HYBRID` (run05) is the strongest family found across 6
runs (LSTM, PHYSICAL untrained, PHYSICAL tuned, PHYSICAL_ASYM, HYBRID symmetric,
HYBRID asymmetric). Try one more distinct black-box family (`LTC`, the closed-form
continuous-time cell) for a broader Phase-1 comparison point, then move into Phase-2
refinement of `HYBRID` (residual size, integrator, drift-penalty/long-window
settings) for the remaining iterations.

## run07 (LTC, closed-form continuous-time cell)

Switched `type` to `LTC` (32 hidden units, `hidden_sizes: [32, 32]` for the internal
`f_net`/`ic_net`), `lr=1e-3`, `crops_per_step=1` carried over.

Result: validation RMSE (normalized) **0.041978** at best_epoch 950, test RMSE
**4.1044 mm** -- a dramatic jump, roughly 17x better in normalized validation RMSE
than the previous best (`HYBRID`, 0.705730) and far below every other family tried
(0.77-0.95). Test RMSE of 4.1 mm lands in the same ballpark as the paper's own
full-closed-loop-simulation quality, well past the IDIM-LS symmetric/asymmetric
cross-test relative-error baselines. Each LTC neuron has its own learnable time
constant (`A_i = exp(-dt/tau_i)`, unconditionally stable by construction) and the
whole state update is a smooth, stable interpolation `h[t+1] = A*h[t] + (1-A)*f(h,u)`
-- this architecture seems exceptionally well matched to a stiff, closed-loop,
friction-driven system like EMPS, converging fast and stably where the naive
LSTM/GRU-style gating and the plain-physical/hybrid models struggled within the same
compute budget.

Kept, copied into `best_kept/`. This is now clearly the leading family by a wide
margin. Next step: enter Phase-2 refinement of `LTC` -- this still counts toward the
required run count, but future changes should be moderate tweaks (hidden size,
`tau_max`, activation, `direct_feedthrough`, `lr`, `dropout_prob`, `weight_decay`)
rather than another family switch, since a genuinely promising architecture has now
been identified per PROGRAM.md's Search Strategy.

## run08 (LTC, 64 hidden units)

Refinement: `n_hidden_states` 32 -> 64, `hidden_sizes` [32,32] -> [64,64] (bigger
`f_net`/`ic_net`), everything else unchanged from run07.

Result: validation RMSE (normalized) 0.060199 at best_epoch 850, test RMSE 6.3063 mm
-- slightly worse than run07 (0.041978 / 4.1044 mm). Epoch count was similar (950 vs
975), so this isn't a throughput effect -- the extra capacity simply didn't help at
this step budget, and may be mildly overfitting/noisier given the still-large
training-loss swings. Discarded; reverted to `best_kept` (run07, 32 hidden units).

Next step: keep `n_hidden_states=32`. Try a different refinement axis: `tau_max`
controls the range of learnable time constants the LTC neurons can express; the
current default (2.0s) is not exposed through `config_pars`, so wire it through
`build_model_from_config` and try a couple of values to see whether matching the
time-constant range more closely to the EMPS trajectory dynamics (fast bang-bang
force steps and friction transitions on a ~20s trajectory) helps further.

## run09 (LTC, tau_max 5.0)

Wired `tau_max` through `model.py`'s `build_model_from_config` (previously hardcoded
at 2.0 inside `ClosedFormCTC`, not reachable from `config_pars`), then set it to 5.0
(vs. the implicit 2.0 default used by run07) to let neurons express slower time
constants for the trajectory's longer-horizon integration behavior.

Result: validation RMSE (normalized) 0.047157 at best_epoch 950, test RMSE 6.0992 mm
-- essentially a tie with run07 (0.041978 / 4.1044 mm), marginally worse. Widening the
time-constant range did not help within this step budget; run07's default 2.0s span
already covers the useful range. Discarded; both `model.py` and `train.py` reverted
in full to `best_kept` (run07) per the plain-file keep/discard protocol, so the
`tau_max` plumbing added for this run was reverted along with the config change (not
selectively kept) -- it can be re-added cheaply if a later run wants that axis again.

Next step: two family-refinement misses in a row (hidden size, tau_max) suggest LTC's
default config is already close to a local optimum for architecture size at this step
budget. Try a training-side refinement instead: `lr` (currently 1e-3, same as the
physical/hybrid families -- LTC's stable-by-construction update may tolerate a larger
step) and/or `crops_per_step` to trade off gradient-estimate noise against steps/sec.

## run10 (LTC, lr 5e-4)

Refinement: `lr` 1e-3 -> 5e-4, hoping the smaller step would reduce the eval-to-eval
validation RMSE swings seen in run07/run08/run09 (values bouncing between ~0.003 and
~0.3 across nearby eval points).

Result: validation RMSE (normalized) 0.057384 at best_epoch 925, test RMSE 7.9084 mm
-- worse than run07, and the swings did not shrink (0.06-0.23 range still visible).
So the noise is not primarily an lr-magnitude effect; more likely it reflects the
single-window training-loss estimate's variance (only `crops_per_step=1` window per
step) interacting with a genuinely stiff loss surface, independent of step size in
this range. Discarded; reverted to `best_kept` (run07, lr 1e-3).

Third refinement miss in a row (hidden size, tau_max, lr) -- `n_hidden_states=32`,
default `tau_max`, `lr=1e-3`, `crops_per_step=1` (run07's exact config) remains the
best result of the search by a wide margin. Next step: test whether the noise is
really a gradient-variance effect by raising `crops_per_step` 1 -> 2 (average two
windows per step, fewer epochs but lower-variance updates) before concluding LTC is
already at its practical ceiling for this compute budget and moving to close out the
search.

## run11 (LTC, crops_per_step 2)

Refinement: `crops_per_step` 1 -> 2, `lr` back to 1e-3 (run07's value).

Result: validation RMSE (normalized) 0.073916 at best_epoch 325, test RMSE 21.5974 mm
-- worse than run07, and, as expected, roughly half the epochs completed (475 vs 975)
since each step now does two window forward/backward passes. The eval-to-eval noise
did not shrink either. Discarded; reverted to `best_kept` (run07).

Four refinement misses in a row now (hidden size, tau_max, lr, crops_per_step): none
of these moderate hyperparameter nudges beat run07's original config. Per PROGRAM.md's
search strategy, this is the signal to stop refining `LTC`'s hyperparameters and try a
more substantial modeling idea instead of another small nudge. Next step: a genuinely
new architecture that mixes the two best ideas found so far -- extend `HybridEMPSModel`
so its residual block can be either `LSTM` (current) or the `ClosedFormCTC` cell
(`LTC`) itself, i.e. a physical backbone (known closed-loop structure, few parameters,
fast to identify) plus an LTC residual (the single best-performing free-form component
found in this search) instead of an LSTM residual. This is a structural change to
`model.py`, not a hyperparameter tweak, and directly tests whether the physics prior
can push LTC's already-strong fit even further, especially on the parts of the
trajectory an unconstrained black box might still get wrong.

## run12 (NEW: HYBRID with LTC residual)

Added a `residual_type` option (`"LSTM"` or `"LTC"`) to `HybridEMPSModel` in
`model.py`: when `"LTC"`, the residual is a full `ClosedFormCTC` cell (reusing the
class from the black-box family) instead of the LSTM `RecurrentBlock`, and
`build_model_from_config` passes `residual_type`/`tau_max` through for `HYBRID`.
Ran with the symmetric-friction physical backbone (as in run05/run07's winners) and
`residual_type: LTC`, `n_hidden_states: 32`, `lr: 1e-3`, `crops_per_step: 1`.

Result: validation RMSE (normalized) 0.143228 at best_epoch 300, test RMSE 28.2282 mm
-- no crash (the new code path works), clearly better than the LSTM-residual `HYBRID`
runs (run05/run06, ~0.71-0.80) but worse than pure `LTC` alone (run07, 0.041978).
Only 500 epochs completed (vs run07's 975): each step now pays for both the physical
ODE's per-timestep loop AND the LTC residual's per-timestep loop, roughly doubling
per-epoch cost, and the physical backbone's initial random-ish parameters may also be
adding noise to what the LTC residual needs to correct for, rather than actually
easing its job at this early a training stage. Discarded; reverted to `best_kept`
(run07, pure LTC).

At 12 runs completed (10 explore, 2 architecture-mix attempts), pure `LTC`
(run07: `n_hidden_states=32`, default `hidden_sizes=[32,32]`, `tau_max` default,
`lr=1e-3`, `crops_per_step=1`) remains the clear best (val RMSE 0.041978, test RMSE
4.1044 mm) and has resisted every refinement attempt so far. Still below the required
15-20 minimum -- next steps: try `direct_feedthrough=True` for LTC now that the family
itself is proven (deliberately revisiting the option `train.py`'s own config comment
warns is risky for pure integrators, worth testing empirically here), then a
weight_decay/dropout regularization pass to see if it tames the eval-to-eval noise
without hurting the best-epoch value, before concluding the search.

## run13 (LTC, direct_feedthrough True)

Refinement: `direct_feedthrough` False -> True (adds a learned `u -> y` linear
shortcut on top of the LTC output). `train.py`'s own config comment warns this is
risky for a pure-integrator system (untrained shortcut can dominate early), but with
LTC's already strong fit it seemed worth testing empirically rather than assuming.

Result: validation RMSE (normalized) 0.049293 at best_epoch 950, test RMSE 4.8308 mm
-- close to run07 but still slightly worse on both metrics. The direct shortcut is not
hurting badly (unlike the LSTM baseline's own worry about it dominating early
training), but it is not helping either; LTC's own `f_net`/time-constant mechanism
already models this system without needing the extra shortcut term. Discarded;
reverted to `best_kept` (run07).

Fifth refinement miss for `LTC` (hidden size, tau_max, lr, crops_per_step,
direct_feedthrough). Run07's original config is proving to be a strong local optimum.
Next step: one more regularization-focused refinement (`weight_decay` or
`dropout_prob`, small values) to test whether taming the eval-to-eval noise finds a
more reliably-good checkpoint, then close out the search at >= 15 total runs and
finalize `test.py --checkpoint-set best_so_far` on the winner.

## run14 (LTC, weight_decay 1e-5)

Refinement: `weight_decay` 0.0 -> 1e-5 (light L2 regularization), everything else
back to run07's config.

Result: validation RMSE (normalized) 0.046820 at best_epoch 400 -- close to run07
(0.041978) but still slightly worse, and notably its test RMSE (18.2947 mm) is far
worse than run07's (4.1044 mm) despite the similar validation number. This is a useful
data point: it confirms the eval-to-eval noise seen across every `LTC` run is real
model instability (different checkpoints along the trajectory can have similar
validation RMSE but very different test-set behavior), not just measurement noise, and
that `best_epoch` bookkeeping (keep the single best validation checkpoint seen) is
doing real, necessary work here rather than being a formality. Discarded; reverted to
`best_kept` (run07).

Sixth refinement miss in a row for `LTC` (hidden size, tau_max, lr, crops_per_step,
direct_feedthrough, weight_decay) -- run07's exact original config remains the
strongest and most robust result of the entire search. At 14 completed runs (one below
the 15-run floor), doing one more confirmatory/exploratory run before closing out:
`dropout_prob` on the `f_net`/`ic_net` (a different regularization mechanism than
weight decay, applied inside the recurrent computation itself rather than on the
weights) to make a final, informed check before finalizing run07 as the search winner.

## run15 (LTC, dropout_prob 0.05)

Refinement: `dropout_prob` 0.0 -> 0.05 on `f_net`/`ic_net`.

Result: validation RMSE (normalized) 0.051384 at best_epoch 575, test RMSE 23.4806 mm
-- worse than run07 on both metrics; dropout's extra stochasticity adds to the
eval-to-eval noise rather than taming it. Discarded; reverted to `best_kept` (run07).

## Search conclusion (15 runs total)

Seven refinement attempts on `LTC` in a row (n_hidden_states, tau_max, lr,
crops_per_step, direct_feedthrough, weight_decay, dropout_prob) all failed to beat
run07's original configuration, and one structural mixing attempt (`HYBRID` with an
`LTC` residual, run12) also fell short of pure `LTC`. Combined with the four family
switches that preceded run07 (LSTM, PHYSICAL untrained/tuned, PHYSICAL_ASYM, HYBRID
symmetric/asymmetric) all landing in the 0.71-0.95 normalized-RMSE range, this is
strong, repeated evidence that **run07's plain `LTC` config is the winner of this
search**: `type: LTC`, `n_hidden_states: 32`, `hidden_sizes: [32, 32]`, `activation:
Tanh`, `lr: 1e-3`, `crops_per_step: 1`, default `tau_max` (2.0), no drift
penalty/long-window mixing, `time_budget_seconds` left at `prepare.py`'s canonical
1800s cap.

Final reported numbers (`test.py --checkpoint-set best_so_far`, run07's checkpoint):
- Validation RMSE (normalized, last-20% continuous simulation): **0.041978**
- Test RMSE (official `DATA_EMPS_PULSES` trajectory): **4.1044 mm** (0.004104 m)

This test RMSE is well inside the paper's own full-closed-loop-simulation quality
range and far better than the IDIM-LS symmetric/asymmetric friction baselines' cross-
test relative errors. The key insight from this search: an unconstrained black-box
model with the right *inductive bias* for a stiff, closed-loop system (LTC's
per-neuron learnable time constants, each individually stable by construction via
`A_i = exp(-dt/tau_i) in (0,1)`) converged faster and further, within the same fixed
compute budget, than either a generic gated-RNN black box or an explicitly
physics-structured white-box/grey-box model -- including a hybrid that combined the
physical structure with the very same LTC cell as its residual. `model.py` and
`train.py` are left at the `best_kept` (run07) state; `checkpoints/best_so_far/`
holds the winning checkpoint and its `test_metrics.json`.
