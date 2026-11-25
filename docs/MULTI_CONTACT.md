# 0.7 Epsilon: simultaneous inelastic contact (experimental)

The new backend solves all declared active contact impulses together at fixed configuration using the full generalized mass matrix. For J in normal/tangent coordinates, W = J M^-1 J^T and u+ = u- + W p. Normal impulses satisfy nonnegativity, nonnegative post-impact normal velocity and complementarity. Tangential impulses obey the Coulomb disk and oppose post-impact slip, or stop tangential velocity within the disk. The nonlinear projection residual is minimized numerically and checked independently (normal complementarity, friction cone, slip direction, generalized momentum, kinetic energy). No position projection. A failed numerical search is not proof of physical impossibility; no uniqueness certificate, and redundant reaction distributions may differ. Stable contact IDs define deterministic ordering. Only perfectly inelastic simultaneous contacts are supported; the existing single-point restitution mode remains available. Limits: 1–16 explicit points, static planes, no mesh or automatic discovery.

## Standalone simulation

`python solver.py simulate body.json scenario.json result.json`

Use schema `kineforge.multi_contact_simulation.v1`, the same initial state, dtSeconds (<= 0.02), steps, externalLoads and actuation fields as the single-point simulation; replace planeContact by planeContacts, an array of objects with id, bone, localPointM, normal, planePointM, mu and optional restitution (must equal zero). Free-root actuation is forbidden. The gap tolerance defaults to 1e-6 m. The output contains ordered contactIds, per-frame gapsM, events with individual impulses and residuals, failure and status. Contact crossings are bracketed with midpoint/end samples, not full CCD; persistent contact uses first-order velocity stepping. Contacts within gap tolerance may be activated slightly early. Newly penetrating points during a support step cause rejection; reduce dt. An exact proof of nonpenetration between samples is not provided. No pose snapping or penetration correction. `CONTACT_SCENARIO_CANDIDATE` means this bounded numerical scenario passed, not an animation certificate.

## Integration with keyframe replay

Add boolean `supportActivationImpulse: true` to `kineforge.tracking.v1` options with the existing supportSchedule. Default false preserves Gamma behavior. At a declared activation, all currently active points are solved simultaneously, including retained supports. Each must match its prescribed anchor within the existing tolerance. The resulting velocities must meet the sticking speed tolerance; otherwise the replay stops with REVIEW_REQUIRED. Reports include activationImpacts; no hidden root forces. This does not implement sliding tracking, automatic activation timing, airborne collision detection, or horizon-wide optimization. External loads and the prescribed initial state remain assumptions. Original author keys and timing are preserved, even if physical replay violates them. Failed or REVIEW_REQUIRED output must not be treated as accepted motion.

## Verification

`python -m unittest discover -s tests -v`

`python examples/multi_contact_benchmark.py multi_contact_benchmark.json`

Synthetic checks: simultaneous symmetric/asymmetric impulse split, friction, redundant and separating contacts, generalized momentum/energy, sampled landing, slip-to-stick, input guards, and scheduled activation with immutable keys. The scheduled ballistic target continues through the ground after activation: the physical replay stops at the support and therefore correctly reports key deviations rather than silently editing timing. No private character or game acceptance in this version.
