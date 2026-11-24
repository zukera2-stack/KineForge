# Roadmap

0.3 — experimental baseline: unified optimization and sampled inverse dynamics. Public synthetic tests are the reproducible baseline; private dataset evaluations are not public fixtures.

0.4 — implemented experimental sampled contact feasibility: solve external force/moment balance with unilateral normal forces and friction constraints; verify static equilibrium, ballistic motion, impossible contacts and force residuals. Integrate as a differentiable or constrained term without changing key timing silently.

0.5 — implemented experimental configurable hinge/ball/fixed DOFs, neutral-relative axes, sampled limits, passive spring-damper laws and gravity-only actuator bounds, with tests for ambiguous pose reconstruction. Contact-corrected loads, other DOFs and coupled forward integration remain future work.

0.6 — implemented explicit bone-local sphere pairs, substep collision objective and conservative clearance gate over declared shortest-arc FK paths. Mesh fit, impacts and environment collision remain future work.

0.7 alpha — dense coupled reference dynamics, explicit scenario integration, analytic independent trajectory holdout and convergence tests implemented. Remaining: keyframe tracking using coupled backend, contact-corrected actuator loads for authored clips, longer independent character holdouts, impact/sliding and scalable performance.

0.7 beta — experimental inverse-dynamics PD trajectory replay with zero floating-base actuation and sampled key/angle gates. Contact-aware tracking, target generation compatible with joint models and optimal control remain open.

0.7 gamma — scheduled sticking contact tracking and coupled reaction/actuator allocation at each sample, with explicit release and zero-speed activation. Partial: automatic contact scheduling, impact/sliding, horizon-wide control optimization and character acceptance remain open.

0.7 delta — implemented single declared point/static plane impulse and first-order sliding/rest support simulation; no simultaneous contacts, full CCD or contact-aware tracking with impacts. Next: multi-contact impulse allocation and integration into physical replay, then whole-horizon control and independent holdouts.

0.8 — stable engine-independent API, packaging and integration adapters. Keep editors and proprietary data separate. No claim of production readiness until validated.
