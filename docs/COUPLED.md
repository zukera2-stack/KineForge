# Coupled rigid-tree reference backend — 0.7 alpha

`coupled.Tree` is a dense, CPU, SI-unit reference implementation. Generalized velocity comprises world root linear and angular velocity (unless the base is fixed), followed by joint rates in the neutral parent basis. Hinge has one degree of freedom, fixed zero, ball three; unspecified joints are unrestricted ball joints. A configuration contains root position and local SO(3) matrices. Offsets and mass properties follow FORMAT.md. The base may be fixed (implicit external support) or floating. No artificial stabilization or hidden root force is applied.

For each segment, recursive forward kinematics produces COM linear and angular Jacobians Jv/Jw, world inertia I, and bias accelerations from moving parent frames. The generalized mass matrix is sum(m Jv^T Jv + Jw^T I Jw). The bias h includes gravity, centripetal and convective accelerations, and gyroscopic moments. Solve M a = tau + passive + applied - h; inverse computes the required actuation for prescribed acceleration and known external loads. Springs and damping use the same neutral-relative coefficients as JOINTS.md. Hinge axes are exact in this backend. A singular or ill-conditioned matrix is rejected, not regularized silently. Angular limits and actuator caps are checked at integration samples; violation rejects rather than producing a hard-stop impulse. Root pose is not restricted by joint limits.

Explicit fixed sticking points can constrain acceleration (J a + bias = 0). The constraint solve uses the mass-matrix Schur complement and minimum-norm reactions for redundancy. Requires initially coincident anchors and nearly zero point velocity; rejects excessive drift, inconsistent constraints, negative normal force, or friction-cone excess. A rejected minimum-norm reaction does NOT prove no other feasible reaction exists. No contact onset, impact, sliding, complementarity, drift projection, environment collision or automatic contact discovery. No collision response; previous collision diagnostics remain separate.

The explicit midpoint integrator advances local rotations with exponential maps, evaluates the coupled dynamics at the midpoint, and checks the end state. Second-order convergence is empirically tested on a planar two-link pendulum, not proved for all SO(3) trajectories. The solver is not symplectic, adaptive or energy-exact. Reports include energy (kinetic, gravity, spring), total COM, linear momentum and angular momentum about COM. Energy is conserved only for conservative unloaded systems; external work and damping change it. Dense matrices scale poorly; no real-time claim.

## CLI

```
python solver.py simulate body.json scenario.json simulation.json
```

Example scenario (floating body initially at rest, no forces other than gravity):

```json
{"schema":"kineforge.simulation.v1","fixedBase":false,"dtSeconds":0.01,"steps":10,"initial":{"rootM":[0,0,3]}}
```

Optional initial `localQuaternionXYZW` [bone][4] and `velocity` [dof]; optional `actuation` [dof], constant in the stated generalized basis. Root actuation requires explicit `allowRootActuation: true`; prefer `externalLoads` with numeric bone index, `localPointM`, `forceN` and `torqueNm` (world vectors). Optional `stickingContacts` entries use numeric `bone`, `localPointM`, `anchorM`, unit `normal`, `mu`. The output lists DOF order, positions, rotations, acceleration, velocity and invariants per time sample. Failed simulation raises an error without writing a result. Original key poses are not modified. The existing `solve` path still optimizes kinematic trajectories with sampled inverse dynamics and earlier gates; coupled tracking, optimal control and contact-corrected actuator demands for authored clips are not integrated yet. This is a partial 0.7 candidate, not a completed physical animation generator.

## Reproduce

`python -m unittest discover -s tests -v`

`python examples/coupled_benchmark.py coupled_benchmark.json`

Independent analytic double-pendulum equations integrated by SciPy provide withheld trajectory angles and COM positions, with step-halving convergence; free-body internal torque verifies recoil and momentum conservation. Other tests cover static sticking, friction refusal, gravity, known point loads, off-neutral axes, singular inertia, passive torques, caps and deterministic simulation. No private character data distributed.
