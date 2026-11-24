# Experimental JSON contract (0.7 alpha)

Coordinates in meters, time in seconds, FPS positive; a rig declares its coordinate convention. Bone names are unique. One root has parent -1; every other parent is an earlier index. 2–256 bones, 2–10000 frames. Input keys are strictly increasing, include both boundaries, and contain one XYZ triplet per bone. Quaternions are XYZW; matrices act on column vectors. Fixed offsets are expressed in the parent's reference basis. Local rotation composition is `G_child = G_parent @ L_child`, and `P_child = P_parent + G_parent @ offset_child`. Root translation is the first joint position.

- Keys (`sam.keyposes.v1`): `rig` (bones, parents, coordinateSystem), `timeline` (start, end, fps), `keys` (frame, positions, optional label). Optional `solverHints.contacts`: bone, start, end, position. These are soft positional constraints, not contact forces.
- Body (`sam.body.experimental.v1`): identical `rig`, `offsets_m`, `neutralLocalQuaternionXYZW`, `primaryAxes`, `observedAngleEnvelopeDeg`, `observedTwistEnvelopeDeg`. Optional key weights, observed angular rates in deg/s, mass fractions, inertia lengths and collisionProxies (two joint indices and minimum separation in meters). Optional `physical`: massKg, centerOfMassLocalM, inertiaCOMKgM2 (symmetric positive semidefinite, physically realizable principal moments), gravityMps2. COM and inertia are in each segment's local reference basis. The values are user-supplied estimates, not learned biological limits.
- Reference: identical `rig`, strictly increasing `phase` spanning 0 to 1, `localQuaternionXYZW` array [sample][bone][4]. Neutral-only references are supported; withheld-frame tests must not provide the hidden reference motion to the solver.
- Solved (`sam.skeletal.solved.v1`): `originalAuthorKeys` exact input keys, `sourceKeys` corrected positions for editor import, `keyframes` all frames, `lockedFrames` indices, `localQuaternionXYZW` per frame, solver diagnostics. Locked frames are editor metadata, not a hard interior key constraint. Endpoints are fixed after initial reconstruction; impossible poses are flagged. Positions alone do not identify all twists or leaf rotations.

Numerical thresholds are explicit and experimental, not guarantees of motion realism. No game-specific assumptions are required by the public kernel.

Optional `solverHints.physicalContacts` and `contactOptions` declare sampled sticking contacts, distinct from legacy positional hints. See [CONTACTS.md](CONTACTS.md) for the complete contract and residual gates.

Optional body `jointModels` declares neutral-relative joint DOFs, limits, passive coefficients and conditional actuator bounds. See [JOINTS.md](JOINTS.md).

Optional body `collisionModel`: explicit bone-local sphere pairs and adaptive interframe clearance bounds. See [COLLISIONS.md](COLLISIONS.md).

Optional explicit forward simulation input and output: [COUPLED.md](COUPLED.md). Separate scenario schema; no change to authored pose JSON.

Optional coupled tracking options and scheduled sticking supports: [TRACKING.md](TRACKING.md), [SUPPORT.md](SUPPORT.md). Author key schema unchanged.

## Contact impulse experiment (0.7 delta)

The `simulate` command also accepts `kineforge.contact_simulation.v1` for one explicit point/static plane. See [IMPACT.md](IMPACT.md) for schema, event outputs and limits. This is separate from scheduled sticking tracking.
