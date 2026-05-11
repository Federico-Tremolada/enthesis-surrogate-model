# Enthesis Surrogate Model  
## Machine Learning-Assisted Design Exploration of Functionally Graded Tendon-to-Bone Interfaces

![Python](https://img.shields.io/badge/Python-Data%20Analysis-blue)
![Abaqus](https://img.shields.io/badge/Abaqus-FEM-lightgrey)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Surrogate%20Model-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange)

---

# Overview

This project extends the previous **Enthesis FEM Model** study, where different functionally graded material (FGM) transition laws were investigated using finite element simulations of the tendon-to-bone interface (enthesis).

The original FEM project demonstrated that graded interfaces can significantly reduce stress concentrations compared to sharp material transitions.  
However, performing large parametric explorations exclusively through FEM simulations quickly becomes computationally inefficient.

The objective of this second project was therefore to develop a **surrogate machine learning model** capable of predicting the mechanical response of graded enthesis configurations directly from design parameters, reducing the need for repeated finite element simulations.

The project combines:

- Biomechanical FEM modeling
- Parametric dataset generation
- Machine learning regression
- Explainable AI (SHAP)
- Design-space exploration
- Engineering-oriented visualization

---

# Connection with the Previous FEM Project

The surrogate model was entirely built upon the dataset generated in the original FEM investigation:

- Sharp interfaces
- Linear graded transitions
- Exponential graded transitions
- Power-law graded transitions

The original simulations were performed in Abaqus and post-processed using Python and MATLAB.

From these simulations, mechanical descriptors and stress metrics were extracted and organized into a structured dataset suitable for machine learning applications.

---

# Project Workflow

```text
Abaqus FEM simulations
        ↓
Python extraction of mechanical metrics
        ↓
Dataset construction
        ↓
Feature engineering
        ↓
Machine learning surrogate model
        ↓
Validation against FEM data
        ↓
SHAP interpretability analysis
        ↓
Design maps and optimization trends
```

---

# Methodology

## 1. Parametric FEM Dataset

A series of finite element models were generated with varying:

- transition law type
- power-law exponent \( n \)
- enthesis length
- number of discretization layers

The simulations provided stress distributions and global mechanical descriptors.

The main target variable used for surrogate modeling was:

- maximum von Mises stress

Additional extracted quantities included:

- interface stress metrics
- stress gradients
- stress curvature
- area-based stress indicators

---

# 2. Machine Learning Surrogate Model

A regression-based surrogate model was trained to predict the mechanical response of the interface directly from design parameters.

Input features included:

- transition law type
- exponent \( n \)
- enthesis length
- number of layers

The objective was not only prediction accuracy, but also:

- interpretability
- engineering usability
- design-space exploration

---

# 3. Validation Against FEM Simulations

The surrogate model predictions were compared against FEM-generated results.

## Validation Plot

![Validation](figures/final/fig_validation_fem_vs_ml.png)

The model captured the general trends of the simulations reasonably well despite the limited dataset size.

The largest discrepancies appeared in configurations with stronger nonlinear behavior and limited sampling density.

---

# 4. Design Space Exploration

One of the main advantages of the surrogate model is the possibility to rapidly explore trends across the design space without rerunning FEM analyses.

## Effect of Enthesis Length

![Length Effect](figures/publication/fig_length_effect_publication.png)

The results suggest that increasing enthesis length generally reduces peak stress for graded configurations, while linear transitions remain less effective.

---

# 5. Design Maps

## Exponent vs Layer Discretization

![Design Map Layers](figures/publication/fig_design_map_layers_publication.png)

## Exponent vs Enthesis Length

![Design Map Length](figures/publication/fig_design_map_length_publication.png)

The design maps reveal a clear optimal region associated with:

- higher power-law exponents
- smoother graded transitions
- increased enthesis length
- finer discretization

These trends are consistent with the original FEM observations.

---

# 6. Explainable AI (SHAP Analysis)

SHAP analysis was used to understand which parameters most strongly influence the surrogate model predictions.

## Feature Importance

![SHAP Importance](figures/final/shap_barplot.png)

The most influential parameters were:

- power-law exponent
- transition law type
- enthesis length

while the number of layers showed a smaller but still measurable contribution.

---

## SHAP Summary Plot

![SHAP Summary](figures/final/shap_summary.png)

The analysis confirmed that:

- higher exponents generally reduce predicted stress
- smoother graded transitions improve mechanical behavior
- sharp interfaces are associated with worse stress distributions

---

# Key Engineering Observations

The project highlighted several important biomechanical trends:

- Functionally graded interfaces reduce stress concentrations compared to sharp transitions.
- Power-law transitions outperform linear graded laws.
- Increasing the enthesis length generally improves stress redistribution.
- Surrogate models can successfully approximate FEM trends with significantly reduced computational cost.
- Explainable AI techniques can provide physically interpretable insights instead of acting as black-box predictors.

---

# Limitations

Several limitations must be acknowledged:

- The dataset size remains relatively small.
- The FEM model is simplified and two-dimensional.
- Material behavior is purely linear elastic.
- The surrogate model is trained only within the explored parameter space.
- Additional FEM simulations would improve model generalization.

Future developments may include:

- nonlinear material behavior
- fracture mechanics
- fatigue modeling
- probabilistic analysis
- topology optimization
- neural-network-based surrogate approaches

---

# Software and Tools

- Abaqus/CAE
- Python
- NumPy
- pandas
- matplotlib
- scikit-learn
- SHAP

---

# Author

**Federico Tremolada**  
Biomedical Engineer — Politecnico di Milano  
Biomechanics • FEM • Biomaterials • Machine Learning • Medical Device R&D
