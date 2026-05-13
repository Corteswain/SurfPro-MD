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
├── data/                 # curated datasets and processed CSV files
├── MD-simulations/       # molecular dynamics setup and analysis workflows
├── surrogate-models/     # machine learning model training and evaluation
└── README.md
```

## Dataset

The SurfPro-MD dataset combines:

Experimental properties (from SurfPro)
- Log of critical micelle concentration pCMC
- Surface tension at CMC ($\gamma_{\mathrm{CMC}}$)
- Surface pressure at CMC ($\Pi_{\mathrm{CMC}}$)
- Maximum surface excess ($\Gamma_{\max}$)
- Minimum molecular area ($A_{\min}$)
- Adsorption efficiency ($pC_{20}$)
 
MD-derived properties (this work)
- Surfactant's diffusion coefficient $D_{\mathrm{MOL}}$ at given concentation
- Solute's diffusion coefficient $D_{\mathrm{SOL}}$ at given concentation
- Viscosity $\eta$ at given concentation
- Surface Tension $\gamma$ at given concentation

Final size:
1436 molecules


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
All Simulations were carried out with GROMACS2024

### Parameterisation
In MD-simulations/parameterise, run start_runs.sh to submit multiple slurm jobs.
Modify it to choose which molecules to parameterise.
This script will take the parameterise.slurm script from MD-simulations/parameterise/Blank/ and submit it with the appropriate parameters for each of them.

### Equilibration
After parameterisation, go to MD-simulations/equilibrations, and run start-runs.sh with the same molecular IDs.
The script will automatically find the parameterised topologies and carry out equilibration simulations with one slurm job per molecule.

### Production runs
For viscosity, surface tension, and diffusion coefficients, perform the same steps in
MD-simulations/viscosity, MD-simulations/tension, and MD-simulations/bulk-properties --- run start-runs.sh in these directories and the script will automatically find the equilibrated systems and submit slurm jobs accordingly.

### Machine Learning
The Training for all models can be found in surrogate-models/surrogate.ipynb
Open and execute the notebook to produce all XGBoost models, analyse the results, and obtain all figures found in the original publication.
All models will be found together in surrogate-models/models.pkl
This file contains a dictionary with "{target}_{test_id}" where test_id ranges from 0 to 4 to denote the different test splits.
Each of these entries in anohter dictionary with the following entries:
            "target": target,               
            "test_id": test_id,
        
            # metrics
            "rmse_ensemble": ensemble_rmse_s,
            "r2_ensemble": ensemble_r2_s,
            "rho_ensemble": ensemble_rho_s,
            "pcc_ensemble": ensemble_pcc_s,
        
            # models
            "fold_models": fold_models,
        
            # data traceability
            "test_indices": test_indices,
            "split_column": split_col0,
        
            # reproducibility
            "feature_cols": features_all,
            "y_test": y_test,
            "y_pred": ensemble_preds,
            "test_mask": test_mask,





