# Implementation Plan

Status: proposed architecture — not yet implemented.

## Work package A — Deterministic mechanics foundation
1. Choose a first lattice class and set conventions for units, specimen size, thickness, and loading.
2. Select constitutive/element model (e.g., truss/beam or continuum). Document what it cannot represent.
3. Implement geometry generation with seeded random perturbations and mesh or graph validation.
4. Implement displacement-controlled quasi-static loading, deterministic solving, and diagnostics.
5. Choose explicit failure law and operational critical fracture criterion **before** reporting fracture metrics.
6. Store intermediate nodal displacement, element forces/stresses, strains, damage status, and boundary reactions.
7. Create force–displacement, nominal stress–strain, deformation, stress and failure figures.
8. Validate with simple analytical examples, mesh refinement, load-step sensitivity, and regression tests.

## Work package B — Experiments and data
- Immutable config + seed + version manifest per run.
- Parameter sweeps over relative density, defects, connectivity, strut orientation, material, and boundary conditions.
- Separate failed/nonconverged jobs; don't impute them as physical fracture observations.
- Track uncertainty across random geometries and out-of-distribution topologies.

## Work package C — Hypothesis discovery
- Proposal agent creates hypotheses with measurable predictions and experiment requests.
- Execution uses only the validated deterministic simulator.
- Analysis agent generates evidence-linked reports with run IDs, comparisons, and negative results.
- Deduplicate or group near-identical hypotheses before rating.

## Work package D — Research taste study
- Fixed candidate pool, predefined rubric, blinded independent ratings by experts and AI.
- Scores in [-1, 1] and ranking ties allowed.
- Pearson / Spearman correlation, confidence intervals and sample sizes.
- Human inter-rater reliability; separate validation of downstream scientific outcomes.
- Evaluate agent consistency across multiple stochastic runs and prompt variants.

## Work package E — Deeper mechanics and writing
- Replication, ablations, robustness, mechanisms, limitations.
- Publish reproducible plot scripts, data provenance, and claims linked to runs.

## First demo acceptance criteria
A single seeded lattice can run end to end from configuration to a mechanical response curve, stress/deformation visualization, run metadata and passing repeatability check. A fracture claim requires a validated failure law; otherwise label the demo **elastic baseline**.

## Principal Risks
- Solver convergence during damage events.
- Mesh dependence and physically unjustified brittle thresholds.
- Confusing nominal strength, crack-initiation load, and fracture toughness.
- Geometry distributions leaking across training and test sets.
- Expert-rating noise and selection bias.
- AI-generated plausible but unsupported narratives.
