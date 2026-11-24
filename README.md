# Cat Shaped Mechanical Metamaterials

This repository now contains the current cat-materials project. Its historical URL retains the name of an earlier chainmail proposal; that proposal is superseded by this research snapshot. Earlier commits remain in history.

**Author:** Henry Yan. **Snapshot:** November 23, 2026.

The goal is to generate diverse, recognizable cat-shaped two-phase materials with requested apparent specimen moduli using conditional flow matching (CFM). Codex assists the researcher with coding, tool execution, analysis and verification. The trained CFM is the generator; numerical FEM and human visual review evaluate its outputs.

## Lab 4 submission

- [Two-page Lab Report 4](docs/lab4/Lab4_Report_Henry_Yan.docx), including AI usage disclosure.
- [Nine-slide demo](docs/lab4/Lab4_Demo_Henry_Yan.pptx), adapted from the supplied course template.
- [Google Slides demo](https://docs.google.com/presentation/d/1Bpnb3tLiahZ7mzwPqjg8RyOGRO3NYjHmIwl6Brcs_Qw/edit).
- [64-step interactive evaluation](evaluation/runs/cfm_v2_step_165000/index_en.html).
- [128-step interactive evaluation](evaluation/runs/cfm_v2_step_165000/index_128steps.html).

Download/clone the repository and open the HTML files locally: GitHub shows their source rather than executing them. These lightweight copies embed the displayed generated/retrieved images. Raw array download links require the full research workspace. The original research outputs remain unchanged locally.

## Pipeline and evidence

Human goals and budgets → Codex orchestration → curated image data and FEM labels → conditional U-Net training → target plus fixed Gaussian noise → binary generation → real FEM → training retrieval → statistics and HTML review.

- Dataset: 7,857 curated binary images in 8 art categories; 6,285/786/786 train/validation/test split. Full image data are omitted from Git; schema, class definitions, index and split files are included.
- Materials: white is soft (E=1), black is hard (E=10), and both have Poisson ratio 0.3. Both are solid phases.
- Mechanics: Python Q4 plane-stress FEM, one element per pixel, strain-controlled finite-specimen tests, apparent Ex/Ey from reaction forces. This is not Abaqus or periodic homogenization.
- Conditions: s = 0.5 ln(Ex Ey / Esoft²), a = ln(Ex/Ey), with Esoft=1.
- Current generator: CFM v2 without D4 augmentation, 11,529,633 parameters; training stopped at 180k under the agreed plateau rule, with the best validation EMA at 165k. This stopping rule is not proof of full convergence.
- Evaluation: 24 frozen target positions and 10 fixed seeds per target. Checkpoints are selected by validation velocity MSE, not FEM error. The fixed panel is reused and is not a fresh holdout.

| Metric over all 240 fixed samples | 64 midpoint steps | 128 midpoint steps |
| --- | ---: | ---: |
| Network evaluations per image | 128 | 256 |
| Valid FEM solutions | 240 | 240 |
| Joint s-a hits | 109 (45.4%) | 106 (44.2%) |
| Both Ex and Ey within 5% | 65 (27.1%) | 69 (28.8%) |
| Mean Euclidean Ex/Ey error | 0.360630 | 0.360589 |

A joint s-a hit requires |delta s| <= ln(1.05) AND |delta a| <= ln(1.10). All seeds count in the denominator; this is distinct from selecting the best of 10. The 128-step run uses the same checkpoint, targets, seeds and initial noise. Four matched examples show no clear visual improvement; this does not establish that additional sampling steps never help.

Recognizable examples include T08/730000 and T04/730007. T12/730000 returns a thin white fragment despite a valid FEM solution. The gold frame in HTML denotes the mechanically best generated sample, not the most cat-like one. Training retrieval is a non-generative baseline and does not test visual memorization. Blinded visual ratings, diversity, novelty, sparse-target generalization and isolated agent-versus-manual productivity remain unresolved.

## Code map

- `mechanics.py`, `mechanics_checks.py`: FEM and physics checks.
- `reference_style_*.py`, other root generation scripts: original data creation/curation workflow.
- `cat_binary_dataset_curated_v1/`: lightweight data documentation and index.
- `cfm_v0/`, `cfm_continuation/`, `cfm_v1/`, `cfm_v1_cont1/`, `cfm_v2/`: model, training, continuation and ablation code.
- `evaluation/`: adapters, fixed target/seed configs, generation, FEM, retrieval, cache checks and reporting.
- `analysis/`: selected distribution plotting scripts.
- `docs/lab4/`: submitted report and demo.

## Reproduce checks and inspect results

Python 3.10 was used for the research environment. The included requirements specify the original CUDA/PyTorch environment; adapt the PyTorch wheel to your hardware before installation if needed.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python evaluation/tests_core.py
python scripts/check_published_results.py
```

The first command runs seven core invariants covering target transforms, training-only retrieval, source uniqueness, cache corruption, fixed distance scales and failure accounting. They were rerun successfully on November 23. The second check uses only the published CSV files and Python's standard library, so it needs neither training data nor a GPU.

To regenerate images or retrain, restore the excluded assets at the paths recorded in `EXTERNAL_ASSETS.json`, and obtain the full curated metadata/images and the run-preparation artifacts referenced by the configurations. The recorded configs intentionally preserve their original data/checkpoint binding. This Git snapshot alone is sufficient for inspecting the saved evidence and testing core logic, but not for reproducing training without those external assets. No public download location for the full assets is claimed.

With the complete research workspace restored:

```powershell
python evaluation/run.py --config evaluation/runs/cfm_v2_step_165000/config.json
python evaluation/resample_v2_165k_128steps.py
```

These commands run inference/evaluation, not training. See `evaluation/README.md` and `cfm_v2/REPORT.md` for stage-level operation, solver conventions, checkpoint selection and stopping policy.

## Progress and next two weeks

Completed: curated dataset, FEM labeling, CFM training/ablations, integrated generation-to-HTML evaluation, controlled 64/128-step sampling comparison, and numerical/provenance checks.

Remaining: define and apply a blind cat-quality rubric and inspect visual nearest neighbors (Nov 23-29); inspect flow-matching trajectories and time-dependent errors, then test one controlled intervention (Nov 30-Dec 6). Freeze evidence and rehearse the demo before the final presentation. The main bottleneck is understanding and improving shape quality.

Risks include subjective visual ratings, memorization, sparse mechanical support and incomplete external-asset availability. Report these limits explicitly; do not infer cat quality from a low mechanics error.

## Publication scope and AI disclosure

This is a curated publication snapshot of the current research workspace. Large training weights, the full image dataset, virtual environments, caches, raw sampling arrays, bulk training logs, archived ZIP packages and duplicate intermediate outputs are intentionally excluded. No historical local research files were deleted to prepare this snapshot. Each file in `PUBLICATION_MANIFEST.json` records its hash and size.

ChatGPT/Codex assisted with code, workflow orchestration, experiment analysis, figures and drafting. Generative image tools created source images. Reported mechanics derive from executed numerical code, and visual/scientific conclusions require human judgment.
