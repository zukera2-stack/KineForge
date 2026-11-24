# Roadmap

0.3 — experimental baseline: unified optimization and sampled inverse dynamics. Public synthetic tests are the reproducible baseline; private dataset evaluations are not public fixtures.

0.4 — implemented experimental sampled contact feasibility: solve external force/moment balance with unilateral normal forces and friction constraints; verify static equilibrium, ballistic motion, impossible contacts and force residuals. Integrate as a differentiable or constrained term without changing key timing silently.

0.5 — joint models: configurable hinge/ball/other DOFs, anatomical axes, explicit limits, spring-damper laws, actuator limits and tests for ambiguous pose reconstruction.

0.6 — collision geometry and self-contact, continuous or substep checks, robust failure reporting.

0.7 — coupled forward dynamics and inverse/forward consistency, convergence and deterministic reproducibility, independent holdout benchmark with error against source poses (not just smoothness).

0.8 — stable engine-independent API, packaging and integration adapters. Keep editors and proprietary data separate. No claim of production readiness until validated.
