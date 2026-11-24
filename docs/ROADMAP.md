# Roadmap

0.3 — experimental baseline: unified optimization and sampled inverse dynamics. Public synthetic tests are the reproducible baseline; private dataset evaluations are not public fixtures.

0.4 — implemented experimental sampled contact feasibility: solve external force/moment balance with unilateral normal forces and friction constraints; verify static equilibrium, ballistic motion, impossible contacts and force residuals. Integrate as a differentiable or constrained term without changing key timing silently.

0.5 — implemented experimental configurable hinge/ball/fixed DOFs, neutral-relative axes, sampled limits, passive spring-damper laws and gravity-only actuator bounds, with tests for ambiguous pose reconstruction. Contact-corrected loads, other DOFs and coupled forward integration remain future work.

0.6 — implemented explicit bone-local sphere pairs, substep collision objective and conservative clearance gate over declared shortest-arc FK paths. Mesh fit, impacts and environment collision remain future work.

0.7 — coupled forward dynamics and inverse/forward consistency, convergence and deterministic reproducibility, independent holdout benchmark with error against source poses (not just smoothness).

0.8 — stable engine-independent API, packaging and integration adapters. Keep editors and proprietary data separate. No claim of production readiness until validated.
