# Collision model (0.6 experimental)

Optional body `collisionModel` supplies an explicit union of spheres attached to bones; distances are meters. Only listed pairs are checked. A missing model means collision checking is not performed. Same-body pairs and duplicate pairs are rejected. Adjacent bodies are not automatically excluded: profile authors must select pairs and avoid intentional overlap. Sphere approximation is not a validated skin or mesh envelope.

```json
{"collisionModel":{"spheres":[{"id":"a","bone":"base","centerLocalM":[1,1,1],"radiusM":0.2},{"id":"b","bone":"tip","centerLocalM":[1,0,0],"radiusM":0.2}],"pairs":[["a","b"]],"toleranceM":0.001,"maxSubsteps":64,"objectiveSubsteps":4}}
```

`maxSubsteps` is a power of two, 2–256; `objectiveSubsteps` 1–16, no greater than max. Default tolerance 1 mm, range 0–50 mm; zero is allowed. Up to 512 spheres and 4096 pairs. Local centers use the same basis as COM. Root interpolation is linear and local rotations use the shortest SO(3) arc, composed by FK. Other engine interpolation is not certified. No independent interpolation of world-space endpoints.

The objective penalizes sampled penetration, including intermediate poses. Its weights are heuristic. Acceptance independently computes a conservative lower bound on clearance over each interval: minimum sampled clearance minus `sum(relative angular travel × maximum lever arm)/(2 × substeps)`. Shared ancestor rotations and root translation do not change pair distance. Dyadic refinement continues for unresolved intervals. Numerical precision is floating point; this is not interval arithmetic. The bound applies only to the declared path. `COLLISION_PENETRATION` indicates a sampled overlap greater than tolerance; `COLLISION_INTERVAL_UNCERTIFIED` means the lower bound cannot exclude such overlap. `COLLISION_ROTATION_AMBIGUOUS` flags relative relevant local rotations near 180°. Any flag requires review. A sample penetration is not necessarily the exact maximum. Report gives pair IDs, sample witness frame (possibly fractional), lower bound, and affected interval start frames.

A declared proxy path PASS is not overall solver acceptance. Original author keys and timing remain unchanged; the solver can flag incompatible boundaries. Zero-distance coincident centers can have zero gradient; no guarantee of repair. No collision forces, impulses, sliding, deformable surfaces, environment obstacles, automatic sphere fit or automatic pair coverage. The legacy collisionProxies field retains its original sampled soft cost only; migrate to collisionModel for an independent gate. Worst-case costs grow with frames, bones, pairs and substeps; production-scale profiling remains future work.

Run `python -m unittest discover -s tests -v` and `python examples/collision_benchmark.py collision_benchmark.json`. Benchmarks include hidden interframe overlap, static penetration, conservative unresolved case, clear path and repair.
