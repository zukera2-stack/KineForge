# Prescribed sticking contacts (0.4, experimental)

The external wrench required by sampled inverse dynamics is computed about the root joint. For each interior sample, active contacts contribute point forces at the current transformed local contact points. Each force is a nonnegative combination of eight rays of an inscribed polygonal Coulomb cone (one normal ray when `mu=0`). This guarantees nonnegative normal force and `|tangent| <= mu * normal`; the polygon is conservative and may reject a solution feasible in the full circular cone. Point contacts transmit no independent torque. Multiple points on a rigid segment can represent a support patch. The contact schedule and friction are **inputs**, never inferred from bone names.

For force `F` and moment `M`, solve a nonnegative least-squares projection in the scaled wrench space `[F, M / momentLengthM]`. The achieved force and moment are subtracted from the requested wrench, with both residuals independently reported and gated in SI units. The solver checks the NNLS optimality conditions; failure raises an error. The force distribution is not unique in general, and the returned distribution is not a muscle or actuator solution. It is not a minimum-effort allocation. During trajectory optimization the converged projection coefficients are held fixed for differentiation of the minimized residual (envelope derivative away from degeneracies). Contact geometry is differentiated through FK. With explicit physical contacts, the gravity-only segment force penalty is disabled because it is not a contact-aware actuator-load model. Spatial and angular regularization and key fidelity remain in the same objective.

## Input

Add to existing keypose `solverHints`:

```json
{
  "physicalContacts": [
    {"id":"support-a", "bone":"left", "start":5, "end":13,
     "localPointM":[0,0,0], "anchorM":[-1,0,0], "normal":[0,0,1], "mu":0.4}
  ],
  "contactOptions": {
    "momentLengthM":1.0, "forceToleranceN":1.0,
    "momentToleranceNm":1.0, "anchorToleranceM":0.005,
    "objectiveWeightM2":0.02, "anchorWeight":2.0
  }
}
```

All option values must be positive and finite; shown defaults apply when omitted. Contact IDs are unique; ranges include both endpoints and use actual frame numbers. `localPointM` is expressed in segment-local coordinates; anchor and unit normal are in world coordinates. `mu` is nonnegative and finite. Mass, gravity and inertia must be supplied in the body profile. `physicalContacts` omitted means support unknown (legacy behavior). An explicit empty list means flight throughout; any uncovered sample in an explicit schedule is also flight. There are no external forces besides gravity and these contacts. For wholly unsupported clips, the initial translation interpolates gravity-compensated key coordinates, then adds the ballistic term back; this does not guarantee ballistic COM when joints move. Existing `contacts` positional hints remain backward-compatible and are not force constraints.

## Output and limits

`report.json` contains `contact_initial`, `contact_final` (and `analyze` returns `contact`). Each includes peak force residual in N, peak moment residual in N·m, maximum anchor gap in m, per-interior-frame target/residual and contact force vectors, flags and numerical settings. Wrench checks exclude the first and last sample (central finite differences); anchors are checked at all active frames. `SAMPLED_CONTACT_FEASIBLE` indicates only the declared point-contact wrench and anchor tolerances passed. `CONTACT_WRENCH_RESIDUAL` and `CONTACT_ANCHOR_DRIFT` require review. `REVIEW_REQUIRED` may also indicate other quality gates or rejected optimization. Key positions and timeline are retained separately; corrected interior positions can differ within configured tolerances, and contradictions are flagged. Contact/key gates are prioritized over smoothness when selecting among candidates; an infeasible result is never silently accepted.

No sliding dynamics, impacts, contact discovery, inactive-point nonpenetration, continuous collision, muscle/actuator limits, force distribution over time, or forward integration. At contact transitions the sampled finite-difference acceleration may span two modes. The report's legacy inverse-dynamics loads are gravity-only, not internal loads corrected for contact reactions. A sampled contact pass is not a proof of continuous motion stability. The numerical defaults are experimental, not physical material properties.

## Reproduction

```sh
python -m unittest discover -s tests -v
python examples/contact_benchmark.py contact_benchmark.json
```

The benchmark uses analytic three-segment models: static off-center two-point support, sparse ballistic flight, unsupported hover, incompatible accelerated sticking motion, and insufficient single-point support. It checks expected pass or rejection, source-key preservation and unchanged timing. No game assets are required.
