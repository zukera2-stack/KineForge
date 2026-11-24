# KineForge

**Universal Physical Motion Solver — experimental 0.7.0-alpha**

KineForge reconstructs and optimizes articulated-body motion from sparse positional keyframes. One numerical objective combines rigid forward kinematics, key fidelity, spatial curve regularity, rotational smoothness, observed joint envelopes and optional sampled inverse dynamics. It is independent of any game, editor or proprietary model.

## Quick start

Python 3.11+; CPU execution. From the repository root:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python examples/generate.py demo_input
python solver.py solve demo_input/keys.json demo_input/body.json demo_input/reference.json demo_output --iterations 120
python solver.py analyze demo_output/solved.json demo_input/body.json analysis.json
```

Output directories and report files must not already exist. `solved.json` preserves the original authored keys separately, `report.json` records quality gates and candidates, and `trajectories.csv` contains joint coordinates, local quaternions and finite-difference diagnostics. When mass properties are provided, `physical_loads.csv` contains sampled wrench estimates. A `REVIEW_REQUIRED` result is not approved motion. `KINEMATIC_CANDIDATE` is not a physical feasibility certificate.

Optional [joint models](docs/JOINTS.md) specify hinge/ball/fixed DOFs, angular limits, passive spring-damper laws and conditional actuator gates. Run `python examples/joint_benchmark.py joint_benchmark.json`.

See [prescribed contact contract](docs/CONTACTS.md) and run `python examples/contact_benchmark.py contact_benchmark.json` for analytic contact checks.

## Scope

- Implemented: ordered single-root tree, fixed bone offsets, SO(3) optimization, spatial and angular trajectory diagnostics, observed rotation/rate envelopes, point-mass inertia validation, gravity-only sampled Newton–Euler inverse dynamics, conservative sampled point-contact feasibility and friction, candidate rejection, configurable joint constraints and gravity-only actuator estimates.
- Experimental: positional contact penalties, sphere-pair collision penalties, profile calibration, inferred twist, heuristic objective weighting.
- Not implemented: sliding/impact dynamics, contact discovery, contact-corrected actuator loads, validated biological anatomy, forward elastic integration, continuous collision detection, coupled keyframe tracking or mesh deformation certification.

The current JSON field names retain legacy `sam.*` schema identifiers for interoperability. See [data contract](docs/FORMAT.md) and [roadmap](docs/ROADMAP.md). The editor is intentionally outside this repository. All examples and public tests use synthetic data; no third-party game assets, private motion datasets or derived character profiles are distributed.

[Русское описание](README_RU.md) · [Contributing](CONTRIBUTING.md) · MIT license (project-owned code only).


## 0.6 collision candidate

Explicit sphere-pair geometry, sampled collision objective and conservative interframe clearance gate. Not full mesh collision or impact dynamics. See [COLLISIONS.md](docs/COLLISIONS.md).

## 0.7 alpha: coupled dynamics

Dense articulated mass matrix and bias forces, forward/inverse dynamics with explicit loads, fixed or floating base, exact hinge/fixed DOFs, sticking-contact reactions, and midpoint simulation. Run `python solver.py simulate body.json scenario.json simulation.json`. See [COUPLED.md](docs/COUPLED.md). The existing keyframe optimizer does not yet use this backend for tracking; the stage is partial.

### 0.7 beta: physical replay
`python solver.py track body.json solved.json options.json tracked.json` runs coupled feedback tracking. See [tracking contract](docs/TRACKING.md). Fixed-base and unsupported floating-base cases only; not contact-aware motion optimization.
