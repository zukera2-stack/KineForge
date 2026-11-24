# KineForge

**Universal Physical Motion Solver — experimental 0.4.0**

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

See [prescribed contact contract](docs/CONTACTS.md) and run `python examples/contact_benchmark.py contact_benchmark.json` for analytic contact checks.

## Scope

- Implemented: ordered single-root tree, fixed bone offsets, SO(3) optimization, spatial and angular trajectory diagnostics, observed rotation/rate envelopes, point-mass inertia validation, gravity-only sampled Newton–Euler inverse dynamics, conservative sampled point-contact feasibility and friction, candidate rejection.
- Experimental: positional contact penalties, sphere-pair collision penalties, profile calibration, inferred twist, heuristic objective weighting.
- Not implemented: sliding/impact dynamics, contact discovery, actuator limits, validated anatomy, elasticity, continuous collision detection, forward dynamics or mesh deformation certification.

The current JSON field names retain legacy `sam.*` schema identifiers for interoperability. See [data contract](docs/FORMAT.md) and [roadmap](docs/ROADMAP.md). The editor is intentionally outside this repository. All examples and public tests use synthetic data; no third-party game assets, private motion datasets or derived character profiles are distributed.

[Русское описание](README_RU.md) · [Contributing](CONTRIBUTING.md) · MIT license (project-owned code only).
