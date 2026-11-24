# 0.7 delta — explicit point/plane impulse experiment

`impact.point_impulse` uses the existing coupled tree and one declared point on a bone against an immovable plane. All quantities are SI. It solves `M (v+ - v-) = J^T p`; normal point velocity after impact is `-e` times the incoming normal component. A separating point receives no impulse. Stick if the required tangential impulse is within the Coulomb cone; otherwise solve for a sliding impulse on the cone boundary opposite post-impact slip using a finite angular root search. Reject unresolved/ambiguous searches, negative normal impulses, excessive residuals and kinetic energy creation. A verified root is not a proof of uniqueness over all directions. The restitution/slip convention is a rigid impact approximation, not a material deformation model. Positions and local rotations are unchanged by the instantaneous impulse. Internal joint and root velocities can change together. Fixed base is an external constraint, so whole-system free momentum conservation does not apply to it.

## Explicit simulation

`python solver.py simulate body.json scenario.json result.json`

```json
{"schema":"kineforge.contact_simulation.v1","dtSeconds":0.01,"steps":30,"initial":{"rootM":[0,0,0.1]},"planeContact":{"bone":0,"localPointM":[0,0,0],"planePointM":[0,0,0],"normal":[0,0,1],"mu":0.5,"restitution":0}}
```

Same body and initial-state convention as COUPLED.md; `bone` is a numeric index. Normal must be unit; `mu` 0–5 and restitution 0–1 are configurable assumptions. `gapToleranceM` defaults to 1e-6 (range 1e-9–1e-3). Step at most 0.02 s, max 10000 steps. Constant actuation and external loads follow existing simulate; floating root actuation is forbidden. Cannot combine with stickingContacts or supportSchedule.

Free motion uses the existing midpoint integrator. A crossing observed at a half or full step is localized by bisection of free integration from the step start; then the impact is applied and remaining time integrated. This is sampled/bracketed event localization, NOT continuous collision detection (an unobserved excursion may be missed). Initial output frame records pre-impact state; impulses are separately logged. At near-zero normal speed within gap tolerance, use a first-order velocity-level step: predict free velocity, apply zero-restitution impulse for closing normal velocity, advance with corrected velocity. Resting support impulses represent a time-step reaction, not instantaneous collisions; their reported kinetic loss is relative to the predicted free velocity, not a separate physical dissipation budget. No position projection, no snapping; excessive penetration or unresolved events return REVIEW_REQUIRED. Small gap errors within tolerance are admitted. Rotating contacts can drift or fail; numerical convergence is not guaranteed. Recorded events preceding a failure may belong to the attempted final step. Output status CONTACT_SCENARIO_CANDIDATE applies only to this single specified point and sampled simulation, not a whole-body certificate.

The original `track`, `solve`, and scheduled support behavior remains unchanged. No simultaneous contacts, contact-point discovery, moving/curved surfaces, self-impact response, integration with tracking, hard joint stop response, or global control optimization. No claim of character or game acceptance.

## Reproduction

`python -m unittest discover -s tests -v`

`python examples/impact_benchmark.py impact_benchmark.json`

The benchmark tests elastic/inelastic vertical drop, sliding to rest, free flight, first-order convergence of sliding position and an articulated impact. Unit tests include analytical off-center impulse, nonzero internal response, stick/slip, separation, invalid parameters, no hidden root drive, determinism and liftoff. Public fixtures are synthetic.
