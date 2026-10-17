# Agentic AI for ML-Based Fracture Mechanics of Chainmail

> Research proposal and project roadmap — **planning stage; simulation and AI workflows are not yet implemented.**

## Overview

This project investigates fracture mechanics in **two-dimensional (2D) cellular, lattice, and chainmail-inspired metamaterials** through an agent-assisted research workflow. Its central premise is to first establish a **deterministic, reproducible simulation infrastructure** and then use an **agentic AI system** to explore scientific hypotheses, prioritize follow-up studies, and compare AI research judgment with human expert judgment.

Although the repository name refers to *chainmail*, the initial research scope is broader: planar cellular/lattice structures. Chainmail-specific contact, interlocking, and large-deformation fracture physics are potential extensions and **should not be assumed to be modeled by the initial lattice solver**.

## Two Research Objectives

1. **Mechanics:** Identify and quantify geometric, topological, and material features that affect the fracture strength and failure behavior of 2D lattice metamaterials.
2. **AI for science:** Compare human experts' judgments with an AI **research-taste agent** in selecting promising research directions, and measure their agreement quantitatively.

## Research Questions

- Which local and global lattice descriptors — e.g., relative density, strut orientation, node connectivity, cell compactness, defect location, and disorder — best explain fracture initiation and strength?
- How do structures with similar effective elastic stiffness differ in damage localization and fracture response?
- Can a reproducible fracture simulator support autonomous generation and testing of scientifically meaningful hypotheses?
- How consistently does an AI agent rank research opportunities compared with human experts, and how well do those rankings predict subsequent scientific value?

## Proposed Research Workflow

### Stage 1 — Build a deterministic fracture simulation pipeline

Use coding agents to develop and maintain a domain-specific simulation and analysis toolkit. The **numerical solver and its outputs must remain deterministic** under fixed parameters, mesh, software version, and random seeds; agent assistance is for software construction and orchestration, not for substituting plausible-looking results for actual calculations.

**Planned capabilities**
- Generate ordered, disordered, and randomized 2D lattices with explicit reproducible seeds.
- Introduce structural features: geometry modifications, missing struts, notches, defects, disorder, and material heterogeneity.
- Define material properties, boundary conditions, and loading protocols; run parameter sweeps and calibrated simulations.
- Solve baseline elasticity and an explicitly specified damage/fracture model (to be selected and validated).
- Determine operational fracture metrics such as first damage event, peak force / critical load, nominal fracture strength, and crack/failure progression. Where appropriate, distinguish these from fracture toughness.
- Export stress–strain or force–displacement curves, deformed configurations, stress/strain/damage fields, failure maps, and diagnostic figures.
- Capture intermediate fields and solver traces for auditing and downstream learning rather than retaining only final scalar metrics.

**Quality gates:** verification against analytical/toy benchmarks, mesh and time/load-step sensitivity, conservation/balance checks as applicable, documented units and constitutive assumptions, run manifests, stable seeds, and regression tests. Failed/nonconverged simulations must be explicitly marked, not silently included as measurements.

**Example proposed output layout**
```text
results/<experiment_id>/<run_id>/
  config.yaml
  metadata.json
  geometry.json
  mesh.*
  fields/
  force_displacement.csv
  stress_strain.csv
  failure_events.csv
  figures/
  diagnostics.json
```

The goal is a focused research tool inspired by commercial simulation platforms, **not** a claim to reproduce their generality, validation, or capabilities.

### Stage 2 — Stochastic agentic scientific discovery

Once the deterministic pipeline meets verification criteria, build a separate stochastic **hypothesis-generation and experiment-planning** workflow.

Agents may propose lattice features, formulate testable hypotheses, request controlled simulation batches, analyze intermediate and final outputs, and produce evidence-linked discovery reports. Every proposed claim must refer to actual simulation run IDs, uncertainty or sensitivity analysis where feasible, and recorded assumptions. Maintain a separation between **agent-generated ideas** and **solver-generated evidence**.

### Stage 3 — Human vs. AI research judgment

Present the **same standardized candidate findings/proposals** to human experts and an AI research-taste agent. Both independently score each item from **−1 to +1**:

- **−1:** not worth pursuing / weak or misleading direction
- **0:** uncertain or neutral research value
- **+1:** highly promising and worth pursuing

Use a common rubric (novelty, physical plausibility, expected insight, feasibility, and reproducibility). Collect anonymized human ratings when possible, record expert count and inter-rater reliability, and avoid leaking human scores into the agent evaluation before scoring.

Compare judgments using **Pearson correlation** for linear agreement and **Spearman rank correlation** for ranking agreement; report sample sizes and confidence intervals, and consider agreement/error measures alongside correlation. Predefine a held-out evaluation set where practical. **Correlation measures agreement, not scientific correctness.** Also assess whether prioritized directions yield validated discoveries.

### Stage 4 — Deepen selected research directions

For hypotheses supported by initial simulation data and expert/agent reviews:

1. Design controlled follow-up experiments and ablations.
2. Perform robustness and sensitivity studies over lattice classes, seeds, load cases, material parameters, and discretization.
3. Investigate mechanisms through intermediate stress, deformation, and damage fields.
4. Replicate promising effects with independent settings or models; record negative and contradictory results.

### Stage 5 — Select, validate, and write the paper

Identify the strongest scientifically defensible findings for an ML-based fracture mechanics paper. Present the mechanical discovery, numerical validation, interpretation, uncertainty, and the human–AI judgment comparison as clearly separated contributions. Build publication-quality plots and tables from versioned scripts, with traceable raw simulation outputs.

## System Architecture (Proposed)

```text
Geometry + materials + loading + seeds
                  |
                  v
    Deterministic simulation pipeline
    geometry -> mesh -> solve -> postprocess
                  |
                  v
    Versioned data, fields, diagnostics, figures
                  |
          +-------+--------+
          |                |
          v                v
   ML/feature study   Discovery agents
                           |
                           v
                 Hypotheses + evidence
                           |
                  +--------+--------+
                  |                 |
                  v                 v
              Human experts    Taste agent
                  |                 |
                  +--------+--------+
                           |
                           v
                 Score / rank / compare
                           |
                           v
                 Follow-up simulations
                           |
                           v
                   Validated paper
```

## Initial Scope and Methodology

- **Primary domain:** 2D cellular/lattice metamaterials, potentially including chainmail-inspired planar architectures.
- **Baseline mechanics:** quasi-static, small-strain elastic truss/beam or continuum representations as appropriate; fracture model and assumptions must be selected explicitly before claiming any failure predictions.
- **Primary labels:** peak force, nominal strength, first-failure displacement/strain, energy absorption, and structural failure progression.
- **Candidate input features:** relative density; strut thickness and orientation distribution; nodal coordination; cell shape/compactness; defects; topology; degree of disorder; material contrast.
- **Research safeguards:** deterministic reruns, holdout topologies rather than only random sample splits, no training/test leakage, clear uncertainty and known applicability limits.

Fracture **strength**, **fracture energy**, and **fracture toughness** are distinct quantities; measured and predicted outputs will be named and defined precisely.

## Related Work and Research Gap

Van ’t Sant et al. [1] introduced forward/inverse neural networks for designing growth-based 2D cellular metamaterials with targeted **anisotropic elastic stiffness**. Luan et al. [2] used interpretable ML to relate morphological descriptors to the **effective elastic properties** of ordered and disordered cellular metamaterials. Xiang et al. [3] combined VAEs and ANNs to decouple structural and material distributions for **stiffness-driven inverse design** in hybrid metamaterials.

Together, these studies establish useful approaches for geometry generation, structure–property prediction, interpretation, and inverse design. **They do not establish a validated ML fracture prediction system.** Fracture depends on localized stress concentration, damage initiation, failure sequences, and evolving crack paths; similar homogenized stiffness need not imply similar fracture strength. This project aims to extend these methodological ideas with reproducible fracture-focused simulations and to evaluate agent-assisted scientific prioritization. This is a gap identified *relative to these three papers*, not a claim that the broader literature lacks ML-based fracture studies.

## Milestones (AM/ME/CE 295 biweekly labs)

| Checkpoint | Target date (2026) | Planned deliverable |
| --- | --- | --- |
| Lab 1 | Oct 18 | Repo, related work, draft proposal, environment plan, running log |
| Lab 2 | Nov 1 | One-page proposal, defined fracture model, dataset/design specifications |
| Lab 3 | Nov 15 | First verifiable end-to-end toy simulation producing a figure |
| Lab 4 | Nov 29 | Integrated generation-to-simulation-to-analysis demo with documented failure modes |
| Lab 5 | TBD (consult course guideline) | Reliability audit, human–AI evaluation prototype, research presentation |

Dates are target checkpoints from the supplied course guideline, not claims that each stage has been completed.

## Repository Layout

```text
README.md                      Project proposal and roadmap
docs/related_work.md           Literature synthesis
docs/implementation_plan.md    Engineering and validation plan
docs/lab_reports/              Biweekly reporting materials
research_log.md                Ongoing done/remaining/bottleneck log
src/                           Planned implementation
configs/                       Planned experiment configurations
experiments/                   Planned experiment scripts/records
results/                       Planned generated outputs
tests/                         Planned verification tests
```

## Current Status

**Proposal and scaffolding only.** No solver, ML model, automatic discovery agent, human study, or scientific fracture result is included at initialization. Issues and planned deliverables should not be mistaken for completed functionality.

## References

1. Van ’t Sant, S., Thakolkaran, P., Martínez, J., & Kumar, S. (2023). *Inverse-designed growth-based cellular metamaterials*. **Mechanics of Materials**, 182, 104668. https://doi.org/10.1016/j.mechmat.2023.104668
2. Luan, S., Chen, E., John, J., & Gaitanaros, S. (2023). *A data-driven framework for structure-property correlation in ordered and disordered cellular metamaterials*. **Science Advances**, 9, eadi1453. https://doi.org/10.1126/sciadv.adi1453
3. Xiang, Y., Hou, J., Chen, X., Tang, K., & Wang, X. (2025). *Decoupled design of hybrid mechanical metamaterials via ensembled deep learning*. **International Journal of Mechanical Sciences**, 301, 110514. https://doi.org/10.1016/j.ijmecsci.2025.110514
