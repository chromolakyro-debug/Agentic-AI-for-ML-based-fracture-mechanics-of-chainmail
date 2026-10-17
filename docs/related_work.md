# Related Work: ML for 2D Cellular Metamaterials

## Scope

The three starting papers study structure–property prediction and inverse design primarily for **effective elastic properties**. None should be cited as direct validation of crack propagation or fracture toughness prediction. The proposed fracture research is an extension.

## Literature Synthesis

### [1] Van ’t Sant et al. (2023) — Inverse-designed growth-based cellular metamaterials

Uses star-shaped growth parameters to define 2D periodic porous architectures, FFT-based numerical homogenization to estimate anisotropic effective stiffness, and coupled forward/inverse neural networks for stiffness-constrained inverse design. Transferable concept: design-to-property surrogate and inverse search; limitation for this project: target response is linear elasticity rather than failure progression.

### [2] Luan et al. (2023) — A data-driven framework for structure-property correlation in ordered and disordered cellular metamaterials

Builds ordered/disordered 2D lattices from tessellations and geometric perturbations, measures morphological descriptors, and applies interpretable regression and SHAP-style feature attribution. Highlights that strut orientation, connectivity, and cell compactness can influence elastic response, beyond simple density-only scaling. Transferable concept: physically meaningful, interpretable feature engineering; proposed extension: test if those descriptors also predict damage initiation and fracture strength, without presuming they do.

### [3] Xiang et al. (2025) — Decoupled design of hybrid mechanical metamaterials via ensembled deep learning

Generates hybrid designs with distinct unit cells and spatial material distributions; combines structure/material VAEs and ANNs for forward elastic stiffness prediction and inverse design. Transferable concept: separate latent representations for geometry and material heterogeneity; limitation: fracture metrics are not directly evaluated.

## Fracture Research Gap

Homogenized stiffness summarizes average elastic response but does not directly capture local stress concentration, defect sensitivity, initiation of damage, propagating failure patterns, or fracture toughness. This project proposes to generate validated fracture-sensitive labels and intermediate fields, then use interpretable ML to test candidate geometry–fracture relationships. This is a **gap relative to the three reviewed studies**, not a complete survey of all fracture ML literature.

## References

1. Van ’t Sant, S., Thakolkaran, P., Martínez, J., & Kumar, S. (2023). *Inverse-designed growth-based cellular metamaterials*. Mechanics of Materials 182, 104668. https://doi.org/10.1016/j.mechmat.2023.104668
2. Luan, S., Chen, E., John, J., & Gaitanaros, S. (2023). *A data-driven framework for structure-property correlation in ordered and disordered cellular metamaterials*. Science Advances 9, eadi1453. https://doi.org/10.1126/sciadv.adi1453
3. Xiang, Y., Hou, J., Chen, X., Tang, K., & Wang, X. (2025). *Decoupled design of hybrid mechanical metamaterials via ensembled deep learning*. International Journal of Mechanical Sciences 301, 110514. https://doi.org/10.1016/j.ijmecsci.2025.110514
