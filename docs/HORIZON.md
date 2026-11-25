# 0.7 Zeta — bounded whole-horizon direct shooting

`python solver.py optimize body.json solved.json options.json result.json`

The `solve` and `track` commands retain their previous contracts. This new mode consumes an existing compatible `sam.skeletal.solved.v1` target, retains `originalAuthorKeys` and timeline, and searches piecewise-constant joint torques over the **whole** horizon. The free root is never actuated; a fixed base is an external fixture. Initial pose is the target's initial pose; initial generalized velocity is explicit if supplied, otherwise the target spline derivative. The initial velocity is not optimized. Known `externalLoads` follow the coupled scenario contract. No editor dependency.

Example options:

```json
{"schema":"kineforge.horizon.v1","fixedBase":true,"controlBlocks":2,"substeps":4,"maxEvaluations":30,"maxRolloutsPerSchedule":200,"maxTorqueNm":50,"maxKeyErrorMm":10,"maxTargetAngleDeg":5,"maxReplayDifferenceMm":1,"maxReplayAngleDeg":0.5,"effortWeight":0.00001,"smoothWeight":0.00001,"orientationWeightMPerRad":0.1,"targetWeight":0.1,"scheduleCandidates":[[]],"supportActivationImpulse":false}
```

Limits: 1–16 bones, 2–25 frames, 0–8 actuated DOFs, 1–8 control blocks (at most the number of intervals), at most 32 torque parameters, 1–16 substeps per frame interval, 1–60 optimizer evaluations (SciPy nfev, excluding numerical Jacobian calls), 2–1000 actual search rollouts per schedule (default 200; final verification is additional), 1–4 explicit schedule alternatives, at most 16 declared contacts per schedule. Body actuator caps restrict the global torque component cap; for a vector actuator a conservative component box of cap/sqrt(DOF) ensures the norm cap. A zero cap requires a different body model, and is rejected rather than bypassed. Defaults are experiment parameters, not anatomical measurements. `controlBlockByInterval` defines exact piecewise-constant control timing. No silent timing changes.

The residual contains position error at original keys, weighted position error across the interpolated target, local SO(3) error, normalized torque effort and differences between adjacent torque blocks. Weights are heuristic; the objective is not physical energy. The optimizer uses bounded finite-difference least squares, initialized from zero and sampled inverse dynamics. Each evaluation independently rolls the coupled dynamics forward. Failed trajectories receive a large residual, and are never selected as complete candidates. This discontinuous penalty can prevent escape from infeasible seeds. A successful local termination is not a global optimality or infeasibility certificate. Control effort and smoothing can be disabled explicitly for parameter-recovery tests.

## Contacts and verification

`scheduleCandidates` enumerates complete owner-supplied alternatives in the same half-open frame format as [SUPPORT.md](SUPPORT.md); an empty list **inside** the outer list means no support. Timing selection is discrete among those alternatives only, not automatic discovery or continuous optimization of activation times. A contact must already meet its anchor; optional inelastic activation uses simultaneous impulses from Epsilon, with no pose snapping. Continued sticking uses the strict minimum-norm reaction in `Tree.forward` (unilateral and friction tests, position 1e-5 m and velocity 1e-4 m/s). This can reject a feasible redundant distribution; there is no contact-aware force allocator within this open-loop search, no sliding tracking or environment collision. Outgoing constraints are evaluated before release.

The best complete numerical search replay for each schedule is evaluated at twice the integration resolution without re-fitting controls. Candidate gates include original key error, local target angle error, step-refinement pose/angle difference, coupled equation residual (generalized units, 1e-6 absolute), joint/actuator limits, sticking constraints and declared export sphere proxy collision. `HORIZON_CANDIDATE` means these bounded numerical checks passed, not general physical or game acceptance. The search prefers a passing schedule, then fewer flags, then lower objective, then input order. Incomplete refined replay may return a complete coarse replay for diagnosis only, marked `REVIEW_REQUIRED`. If no complete replay exists, `solved` is null. Missing geometry/contact candidates cannot be certified. No continuous interframe collision or hard joint stop response. `frames` contain sampled velocities, torques, contact reactions and invariants; activation impulses are recorded separately. Initial state, loads and schedules remain assumptions.

## Reproduce

`python -m unittest discover -s tests -v`

`python examples/horizon_benchmark.py horizon_benchmark.json`

Independent analytic two-link ODE holdout: three key poses, explicitly known initial velocity, one constant torque block. Six hidden samples excluded from the fitting residual (target and orientation weights zero), compared to independent high-accuracy integration. The recovered torque is not uniquely identified by position keys; do not infer anatomical muscle forces from this test. Further cases: discrete landing schedule selection, no-root-force ballistic flight, impossible hover, actuator cap rejection, budget and refinement gates. Large rigs, private character holdouts and game validation remain outside this release.
