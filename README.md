# SurfPro-MD

Machine learning and molecular dynamics workflows for surfactant property prediction.

## Overview

SurfPro-MD is an extended surfactant property dataset and associated computational workflow combining curated experimental data with molecular dynamics (MD)-derived physicochemical properties.

The repository accompanies the manuscript:

**A unified experimental-simulation dataset and surrogate models for surfactant property prediction]**  
**Richard Beckmann, Pablo Martinez Crespo, Robert S. Jordan, Marisa Gliege, Santiago Miret, Vijay Kris Narasimhan, and Rocío Mercado**  
**https://chemrxiv.org/doi/full/10.26434/chemrxiv.15002790/v1**

The work extends the SurfPro database by augmenting experimentally curated surfactant data with four additional properties derived from atomistic molecular dynamics simulations:

- solute self-diffusion coefficient ($D_{\mathrm{MOL}}$)
- solvent self-diffusion coefficient ($D_{\mathrm{SOL}}$)
- bulk shear viscosity ($\eta$)
- air–water surface tension ($\gamma$)

The final SurfPro-MD dataset contains these 4 MD-derived properties alongside the original 6 experimental properties for 1436 unique surfactant molecules


The goal of this repository is to provide:

- the curated SurfPro-MD dataset
- all simulation workflows
- all machine learning scripts
- trained surrogate models
- analysis scripts used in the manuscript

to enable **full reproducibility**.

---

# Repository structure

```text
SurfPro-MD/
├── data/               # curated datasets and processed CSV files
├── seedfiles/          # input SMILES and seed structures
├── scripts/            # general preprocessing scripts
├── MD/                 # molecular dynamics setup and analysis workflows
├── ML/                 # machine learning model training and evaluation
├── figures/            # figure-generation scripts used in manuscript
├── models/             # saved trained models
├── reference/          # experimental validation datasets
└── README.md
```

## Dataset

The SurfPro-MD dataset combines:
```text
Experimental properties (from SurfPro)
Log of critical micelle concentration pCMC
Surface tension at CMC ($\gamma_{\mathrm{CMC}}$)
Surface pressure at CMC ($\Pi_{\mathrm{CMC}}$)
Maximum surface excess ($\Gamma_{\max}$)
Minimum molecular area ($A_{\min}$)
Adsorption efficiency (pC$_{20}$)
MD-derived properties (this work)
Surfactant's diffusion coefficient $D_{\mathrm{MOL}}$ at given concentation
Solute's diffusion coefficient $D_{\mathrm{SOL}}$ at given concentation
Viscosity $\eta$ at given concentation
Surface Tension $\gamma$ at given concentation

Final size:
1436 molecules
```

# Usage

### Main Python packages:
numpy
pandas
matplotlib
scikit-learn
xgboost
rdkit
optuna
tqdm

### External software dependencies
Open Babel 3.0.1
AmberTools
ACPYPE
GROMACS 2024.3

## Workflow 
Following soon...
