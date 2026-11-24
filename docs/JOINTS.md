# Optional joint models (0.5)

The body passport may contain `jointModels`, an object keyed by **non-root bone name**. The root is free and unlisted joints retain the previous experimental observed-envelope behavior. All rotations are measured relative to that bone's neutral local rotation: `Q = neutral.T @ L`. Axes are unit vectors in this neutral frame, not world axes or the outgoing bone direction unless explicitly equal. This is a sampled soft constraint in the unified objective, followed by an explicit acceptance gate, not a new independent solver.

```json
{
  "jointModels": {
    "elbow": {
      "type": "hinge", "axis": [0, 1, 0],
      "minDeg": -10, "maxDeg": 120, "restDeg": 0,
      "toleranceDeg": 0.5,
      "stiffnessNmPerRad": 12, "dampingNmsPerRad": 2,
      "actuatorLimitNm": 40
    },
    "shoulder": {"type": "ball", "maxAngleDeg": 100},
    "weld": {"type": "fixed"}
  }
}
```

All numbers above are synthetic examples, not biological parameters. Hinge angle is the principal signed twist about the declared axis; the swing error is the angle between that axis and its rotated image. `minDeg`/`maxDeg` in [-179,179], rest within range (default 0); both swing and range excess must be within tolerance. Ball model has a symmetric angular cone about neutral (not an anatomical swing-twist ellipse), `maxAngleDeg` in [0,179]. Fixed model has zero relative angle. `toleranceDeg` defaults to 0.5 (0–10 supported). Near-antipodal chart singularities, twist wrap and >170 degree frame steps are flagged. No multi-turn joints. Unspecified stiffness/damping are zero; unspecified actuator bound disables the actuator gate. Fixed joints cannot specify nonzero spring/damper or an actuator. Invalid/unknown fields fail validation.

## Passive response and actuator estimate

SI: angular displacement radians, angular velocity rad/s, stiffness N*m/rad, damping N*m*s/rad. Hinge passive torque is `-k*(theta-rest)*axis - c*omega_axis`; isotropic ball passive torque is `-k*Log(Q) - c*omega`. The stored elastic energy is `k*|displacement|^2/2` joules. Angular velocity uses sampled relative rotation increments, centered over adjacent intervals. These laws only have their intended physical interpretation inside the declared joint range and near the admissible DOF. Endpoints have no centered torque estimate. For ball joints the isotropic spring uses the principal rotation chart.

When a finite actuator bound is present, the gravity-only Newton–Euler estimate is transformed from world coordinates into the neutral joint frame; for hinges only the allowed axis is an actuator DOF (off-axis reactions are not motor torque). Required actuator torque = estimated required torque minus passive torque. Its norm must not exceed the declared bound. The resulting excess enters the common optimization objective. With no mass model, fewer than three frames, or any prescribed nonempty support schedule, actuator demand is unavailable and `ACTUATOR_LOAD_UNVERIFIED` is mandatory. The gravity-only estimate assumes any additional external wrench is applied at the root, and is not a guarantee for unspecified real-world supports. Springs without a specified actuator bound are reported, not independently integrated or used as an arbitrary smoothing penalty.

## Gates and diagnostics

`joint_initial`, `joint_final` in solve report, `joints` in analyze report: per-joint peak angle error, offending frame numbers, ambiguous frames, passive peak torque, peak spring energy, actuator demand/limit and over-limit frames. `JOINT_RANGE_OR_AXIS_LIMIT`, `JOINT_ROTATION_AMBIGUOUS`, `ACTUATOR_LIMIT`, `ACTUATOR_LOAD_UNVERIFIED` cause `REVIEW_REQUIRED`. Endpoints are still fixed after reconstruction; original keys and time are retained. Gates apply at sampled frames only. No joint limit reaction impulses, contact-corrected internal loads, anisotropic stiffness, continuous collision or coupled forward dynamics yet. `SAMPLED_JOINT_CONSTRAINTS_PASS` is local to the configured constraints, not overall motion acceptance. Observed canonical envelopes and hinge-fit axes are statistical suggestions, not proof of anatomy or measured stiffness.

Run `python examples/joint_benchmark.py joint_benchmark.json` for reproducible synthetic checks and `python -m unittest discover -s tests -v` for regression tests.
