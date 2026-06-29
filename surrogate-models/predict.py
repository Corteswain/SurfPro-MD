#!/usr/bin/env python3
"""
predict.py — SurfPro-MD surrogate model predictor

Predict surfactant properties from a SMILES string using the XGBoost ensemble
models trained in surrogate.ipynb. Each property is predicted by averaging
over a 5-fold cross-validation ensemble, with optional uncertainty estimates
across 5 outer test splits.

Predicted properties
--------------------
  pCMC             — log critical micelle concentration
  AW_ST_CMC        — air-water surface tension at the CMC
  Gamma_max        — maximum surface excess
  Area_min         — minimum area per molecule
  pC20             — efficiency of surface tension reduction
  D_MOL            — molecular diffusion coefficient
  D_SOL            — solution diffusion coefficient
  surface_tension_avg — average surface tension
  viscosity        — dynamic viscosity

Usage (CLI)
-----------
  python predict.py "CCCCCCCCCC(=O)O"
  python predict.py "CCCCCCCCCC(=O)O" --uncertainty
  python predict.py "CCCCCCCCCC(=O)O" --explain
  python predict.py "CCCCCCCCCC(=O)O" --model path/to/models.pkl
  python predict.py "CCCCCCCCCC(=O)O" --training-csv ../data/SurfPro-MD.csv

Usage (Python API)
------------------
  from predict import SurfProPredictor, ApplicabilityDomain

  predictor = SurfProPredictor("models.pkl")

  # Point prediction (single test split, all targets)
  result = predictor.predict("CCCCCCCCCC(=O)O")
  print(result["prediction"])

  # With uncertainty across test splits
  result = predictor.predict("CCCCCCCCCC(=O)O", return_ci=True)
  print(result["uncertainty"])

  # With SHAP feature importances (requires the shap package)
  result = predictor.predict("CCCCCCCCCC(=O)O", return_explanation=True)

  # Applicability domain check
  ad = ApplicabilityDomain(predictor)
  ad.fit_from_csv("../data/SurfPro-MD.csv")
  print(ad.baseline_stats)         # NN-distance distribution within the training set
  print(ad.query("CCCCCCCCCC(=O)O"))  # how far is this molecule from training data?

Dependencies
------------
  rdkit, numpy, pandas, xgboost, scikit-learn
  shap  (optional, only needed for --explain / return_explanation=True)
"""

import argparse
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


# ── Module-level descriptor calculator (built once at import time) ────────────

_DESCRIPTOR_NAMES = [name for name, _ in Descriptors._descList]
_CALCULATOR = MoleculeDescriptors.MolecularDescriptorCalculator(_DESCRIPTOR_NAMES)

# Short, human-readable display names for each target.
# These are used as index labels in returned DataFrames.
# For matplotlib use, you can pass LaTeX strings via the target_labels argument.
DEFAULT_LABELS = {
    "pCMC":               "pCMC",
    "AW_ST_CMC":          "gamma_CMC",
    "Gamma_max":          "Gamma_max",
    "pC20":               "pC20",
    "Pi_CMC":             "Pi_CMC",
    "Area_min":           "A_min",
    "viscosity":          "eta",
    "D_MOL":              "D_MOL",
    "D_SOL":              "D_SOL",
    "surface_tension_avg": "gamma",
}

# Physical units for each target (plain text, for CLI output).
# Sourced from the unit_labels dict in surrogate.ipynb.
UNITS = {
    "pCMC":               "log(M)",
    "AW_ST_CMC":          "mN/m",
    "Gamma_max":          "μmol/m²",
    "pC20":               "log(M)",
    "Pi_CMC":             "mN/m",
    "Area_min":           "nm²",
    "viscosity":          "mPa·s",   # raw model output is Pa·s; see DISPLAY_SCALE
    "D_MOL":              "m²/s",
    "D_SOL":              "m²/s",
    "surface_tension_avg": "mN/m",
}

# Multiply raw model output by this factor before reporting.
# Used to convert to more human-readable magnitudes.
DISPLAY_SCALE = {
    "viscosity": 1e3,   # Pa·s → mPa·s
}


# ── SMILES utilities ──────────────────────────────────────────────────────────

def canonicalize_smiles(smiles: str, keep_largest: bool = True) -> str:
    """Return canonical SMILES, optionally keeping only the largest fragment.

    Parameters
    ----------
    smiles : str
        Input SMILES (may be non-canonical or contain salts/counterions).
    keep_largest : bool
        When the molecule contains multiple disconnected fragments (e.g. a salt),
        keep only the fragment with the most heavy atoms. Default True.

    Raises
    ------
    ValueError
        If RDKit cannot parse the SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    if keep_largest and "." in smiles:
        fragments = Chem.GetMolFrags(mol, asMols=True)
        mol = max(fragments, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol, canonical=True)


def compute_rdkit_descriptors(smiles: str) -> dict:
    """Compute the full set of RDKit molecular descriptors for one SMILES string.

    Returns a dict keyed as ``"rdkit-<DescriptorName>"`` to match the feature
    schema stored inside models.pkl.
    """
    mol = Chem.MolFromSmiles(smiles)
    values = _CALCULATOR.CalcDescriptors(mol)
    return {f"rdkit-{name}": val for name, val in zip(_DESCRIPTOR_NAMES, values)}


# ── SHAP compatibility helper ─────────────────────────────────────────────────

def _make_shap_tree_explainer(booster):
    """Create a shap.TreeExplainer, working around the XGBoost 2.x / SHAP version mismatch.

    SHAP reads the model via booster.save_raw(raw_format="ubj"), decodes the UBJSON,
    and calls float() on learner_model_param["base_score"].  XGBoost >= 2.0 stores
    that value as a bracket-quoted string (e.g. b'[3.2084293E0]') in the binary,
    which float() cannot parse.  load_config() does not fix it — XGBoost reconverts
    the value to bracket notation internally.

    The fix: temporarily replace save_raw() on the booster instance with a version
    that patches the bracket-notation bytes in-place before returning them.  UBJSON
    strings are length-prefixed, so the replacement is padded to the same byte length
    with trailing spaces (Python's float() strips whitespace, so this is safe).
    """
    import re
    import shap

    _orig_save_raw = booster.save_raw

    def _fixed_save_raw(raw_format="ubj"):
        raw = _orig_save_raw(raw_format=raw_format)
        if raw_format == "ubj":
            def _unwrap(m):
                inner = m.group(1)             # e.g. b'3.2084293E0'
                val = float(inner)
                fixed = f"{val:.10g}".encode()
                if len(fixed) < len(m.group(0)):
                    fixed = fixed + b" " * (len(m.group(0)) - len(fixed))
                return fixed[: len(m.group(0))]  # never exceed original length
            raw = re.sub(rb"\[(-?[\d.]+(?:[Ee][+-]?\d+)?)\]", _unwrap, raw)
        return raw

    booster.save_raw = _fixed_save_raw
    try:
        explainer = shap.TreeExplainer(booster)
    finally:
        booster.save_raw = _orig_save_raw  # always restore

    return explainer


# ── Predictor class ───────────────────────────────────────────────────────────

class SurfProPredictor:
    """Ensemble XGBoost predictor for surfactant properties.

    Wraps the ``results`` dict produced by surrogate.ipynb.  Each target has
    N_TEST_SPLITS outer test splits, each containing N_FOLDS fold models.
    Predictions are the mean over all folds; uncertainty is estimated as the
    standard deviation over outer test splits.

    Parameters
    ----------
    model_file : str
        Path to the pickled model file generated by surrogate.ipynb.
    target_labels : dict, optional
        Override the display names used as DataFrame index labels.
        Keys must match the internal target names (e.g. "pCMC", "AW_ST_CMC").
        If omitted, DEFAULT_LABELS is used.

    Examples
    --------
    >>> predictor = SurfProPredictor("models.pkl")
    >>> result = predictor.predict("CCCCCCCCCC(=O)O")
    >>> print(result["prediction"])
    """

    def __init__(self, model_file: str = "models.pkl", target_labels: dict = None):
        with open(model_file, "rb") as f:
            self._results = pickle.load(f)

        self.targets = list(self._results.keys())
        self.target_labels = target_labels if target_labels is not None else DEFAULT_LABELS.copy()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_features(self, smiles: str, feature_schema: list) -> pd.DataFrame:
        """Canonicalize SMILES and return a DataFrame aligned to feature_schema."""
        canonical = canonicalize_smiles(smiles)
        descriptors = compute_rdkit_descriptors(canonical)
        return pd.DataFrame([descriptors]).reindex(columns=feature_schema)

    def _label(self, target: str) -> str:
        return self.target_labels.get(target, target)

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict_target(self, smiles: str, target: str) -> tuple:
        """Predict a single property using all 25 models (5 splits × 5 folds).

        Parameters
        ----------
        smiles : str
            Input molecule.
        target : str
            Internal target name (must be in ``self.targets``).

        Returns
        -------
        (mean, std) : tuple of float
            Mean across all 25 predictions.
            Std is the standard deviation of the 5 per-split means — the 5 folds
            within a split share the same data partition and are correlated, so
            pooling all 25 would overstate the uncertainty.
        """
        # Feature schema is identical across splits — build X once
        first_info = next(iter(self._results[target].values()))
        X = self._build_features(smiles, first_info["feature_schema"])

        split_means = []
        for info in self._results[target].values():
            fold_preds = [
                fold["model"].inplace_predict(fold["scaler"].transform(X))[0]
                for fold in info["fold_models"]
            ]
            split_means.append(np.mean(fold_preds))

        split_means = np.array(split_means)
        return float(np.mean(split_means)), float(np.std(split_means))

    def predict(
        self,
        smiles: str,
        return_ci: bool = False,
        return_explanation: bool = False,
        shap_max_folds: int = 3,
        ad: "ApplicabilityDomain" = None,
    ) -> dict:
        """Predict all surfactant properties for one molecule.

        Uses all 25 models (5 outer splits × 5 CV folds).  The reported std
        is the standard deviation across the 5 per-split means.

        Parameters
        ----------
        smiles : str
            Input molecule as a SMILES string.
        return_ci : bool
            If True, also include min/max across outer test splits in the
            output (in addition to the mean and std that are always returned).
        return_explanation : bool
            If True, compute SHAP feature importances for each target.
            Requires the ``shap`` package and may be slow.
        shap_max_folds : int
            Number of CV folds to use per target when computing SHAP values.
            Fewer folds is faster at some cost to precision.
        ad : ApplicabilityDomain, optional
            If provided, add per-target applicability-domain distances as an
            additional column (``nn_dist_norm``) in the prediction DataFrame.

        Returns
        -------
        dict
            Always contains:
              ``"prediction"`` — pd.DataFrame with columns ``prediction``, ``std``,
              and optionally ``nn_dist_norm``, indexed by display name.
            Optionally contains:
              ``"uncertainty"`` — pd.DataFrame (mean/std/min/max across test splits).
              ``"explanation"`` — dict mapping target label → top-5 SHAP feature dict.
        """
        rows = {}
        for target in self.targets:
            mean, std = self.predict_target(smiles, target)
            scale = DISPLAY_SCALE.get(target, 1.0)
            rows[self._label(target)] = {
                "prediction": mean * scale,
                "std":        std  * scale,
                "unit":       UNITS.get(target, ""),
            }

        df_pred = pd.DataFrame(rows).T

        if ad is not None:
            ad_results = ad.query(smiles)
            df_pred["nn_dist_norm"] = [
                ad_results.get(t, {}).get("nn_distance_normalized", float("nan"))
                for t in self.targets
            ]

        output = {"prediction": df_pred}

        if return_ci:
            ci = self.predict_with_error_bars(smiles)
            ci.index = [self._label(t) for t in ci.index]
            output["uncertainty"] = ci

        if return_explanation:
            explanations = {}
            for target in self.targets:
                mean_imp, std_imp = self.explain_target(
                    smiles, target, max_folds=shap_max_folds
                )
                top = mean_imp.head(5)
                explanations[self._label(target)] = {
                    "top_features": top,
                    "top_feature_uncertainty": std_imp.loc[top.index],
                }
            output["explanation"] = explanations

        return output

    def predict_with_error_bars(self, smiles: str) -> pd.DataFrame:
        """Predict all targets with per-split breakdown (mean, std, min, max).

        Calls predict_target for each target (which already uses all splits) and
        additionally computes the per-split min/max for a full range estimate.

        Returns
        -------
        pd.DataFrame
            Indexed by internal target name, columns: mean, std, min, max.
        """
        per_target = {}

        for target in self.targets:
            first_info = next(iter(self._results[target].values()))
            X = self._build_features(smiles, first_info["feature_schema"])

            split_means = []
            for info in self._results[target].values():
                fold_preds = [
                    fold["model"].inplace_predict(fold["scaler"].transform(X))[0]
                    for fold in info["fold_models"]
                ]
                split_means.append(np.mean(fold_preds))

            split_means = np.array(split_means)
            per_target[target] = {
                "mean": float(np.mean(split_means)),
                "std":  float(np.std(split_means)),
                "min":  float(np.min(split_means)),
                "max":  float(np.max(split_means)),
            }

        return pd.DataFrame(per_target).T

    # ── SHAP explanations ─────────────────────────────────────────────────────

    def explain_target(
        self,
        smiles: str,
        target: str,
        test_id: int = 0,
        fold_ids: list = None,
        max_folds: int = 3,
        return_std: bool = True,
    ):
        """Compute mean absolute SHAP values for one target property.

        SHAP values are computed for each requested CV fold and then averaged.
        This gives a per-feature importance score for the specific input molecule.

        Parameters
        ----------
        smiles : str
            Input molecule.
        target : str
            Internal target name.
        test_id : int
            Outer test split to use (0–4).
        fold_ids : list of int, optional
            Specific fold indices to evaluate. Defaults to the first `max_folds` folds.
        max_folds : int
            Upper limit on the number of folds to evaluate (fewer = faster).
        return_std : bool
            If True, return ``(mean_series, std_series)``; otherwise return ``mean_series``.

        Returns
        -------
        pd.Series or (pd.Series, pd.Series)
            Feature importances sorted descending by mean absolute SHAP value.
            If ``return_std=True``, also returns the per-feature standard deviation.

        Notes
        -----
        Requires the ``shap`` package. The models are stored as ``xgb.Booster``
        objects, so ``shap.TreeExplainer`` is used directly on the stored booster.
        If you encounter version-compatibility errors between SHAP and XGBoost,
        upgrade SHAP (``pip install --upgrade shap``).
        """
        import shap

        info = self._results[target][test_id]
        X = self._build_features(smiles, info["feature_schema"])

        selected = (fold_ids or list(range(len(info["fold_models"]))))[:max_folds]

        shap_values_all = []
        for fold_id in selected:
            fold = info["fold_models"][fold_id]
            X_scaled = fold["scaler"].transform(X)
            explainer = _make_shap_tree_explainer(fold["model"])
            shap_vals = explainer.shap_values(X_scaled)[0]
            shap_values_all.append(np.abs(shap_vals))

        shap_arr = np.array(shap_values_all)
        mean_imp = (
            pd.Series(shap_arr.mean(axis=0), index=info["feature_schema"])
            .sort_values(ascending=False)
        )

        if return_std:
            std_imp = pd.Series(shap_arr.std(axis=0), index=info["feature_schema"]).loc[mean_imp.index]
            return mean_imp, std_imp

        return mean_imp

    def explain(self, smiles: str, max_folds: int = 3) -> dict:
        """Return top-5 SHAP importances for all targets.

        Uses split 0 for SHAP computation (running all 25 models would be
        25× slower with negligible benefit for feature attribution).

        Returns
        -------
        dict
            Maps internal target name → dict with keys
            ``"mean_importance"`` and ``"std_importance"`` (both pd.Series).
        """
        results = {}
        for target in self.targets:
            mean_imp, std_imp = self.explain_target(
                smiles, target, test_id=0, max_folds=max_folds
            )
            results[target] = {
                "mean_importance": mean_imp.head(5),
                "std_importance":  std_imp.head(5),
            }
        return results

    def print_explanation(self, smiles: str, max_folds: int = 3):
        """Print a formatted SHAP feature importance summary for all targets."""
        for target, res in self.explain(smiles, max_folds=max_folds).items():
            label = self._label(target)
            print(f"\n{'='*60}")
            print(label)
            print("=" * 60)
            for feat in res["mean_importance"].index:
                mean = res["mean_importance"][feat]
                std  = res["std_importance"][feat]
                print(f"  {feat:40s}  {mean:.4f} ± {std:.4f}")


# ── Applicability domain ──────────────────────────────────────────────────────

class ApplicabilityDomain:
    """Per-target nearest-neighbour applicability domain in scaled RDKit feature space.

    Each predicted property is trained on a different subset of molecules (not
    every molecule has experimental data for every property).  This class fits
    a separate training-set feature matrix per target so that the domain check
    reflects the actual chemical space each model has seen.

    Approach
    --------
    1. For each target, collect the molecules that have a measured value.
    2. Compute RDKit descriptors and scale with the same StandardScaler used
       during model training.
    3. For a query molecule, find its Euclidean distance to the nearest
       training neighbour in that scaled space.
    4. Normalise by the mean of the within-training pairwise NN distances.
       A normalised value >> 1 signals extrapolation for that target.

    Parameters
    ----------
    predictor : SurfProPredictor
        A loaded predictor instance.

    Examples
    --------
    >>> predictor = SurfProPredictor("models.pkl")
    >>> ad = ApplicabilityDomain(predictor)
    >>> ad.fit_from_csv("../data/SurfPro-MD.csv")
    >>> print(ad.baseline_stats)       # per-target NN-distance distributions
    >>> print(ad.query("CCCCCCCCCC(=O)O"))  # per-target AD distances
    """

    def __init__(self, predictor: SurfProPredictor):
        ref_info = predictor._results[predictor.targets[0]][0]
        self._predictor = predictor
        self._feature_schema = ref_info["feature_schema"]
        self._scaler = ref_info["fold_models"][0]["scaler"]
        self._fill_values = self._scaler.mean_  # impute NaN → 0 after scaling
        self._per_target: dict = {}

    # ── Internal helper ───────────────────────────────────────────────────────

    def _scale_smiles_list(self, smiles_list: list) -> np.ndarray:
        """Return scaled feature matrix for a list of SMILES, skipping invalid ones."""
        features = []
        for smi in smiles_list:
            try:
                X = self._predictor._build_features(smi, self._feature_schema)
                features.append(X.values[0])
            except Exception:
                continue
        if not features:
            return np.empty((0, len(self._feature_schema)))
        arr = np.array(features, dtype=float)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            arr[nan_mask] = np.take(self._fill_values, np.where(nan_mask)[1])
        return self._scaler.transform(arr)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, smiles_col: str = "SMILES") -> "ApplicabilityDomain":
        """Fit per-target training sets from a DataFrame.

        For each target, only rows with a non-NaN value for that target column
        are included.  This mirrors the actual training set each model saw.

        Parameters
        ----------
        df : pd.DataFrame
            The full dataset (same file used by surrogate.ipynb).  Must contain
            a SMILES column and one column per predicted target.
        smiles_col : str
            Name of the SMILES column (default: ``"SMILES"``).

        Returns
        -------
        self
        """
        from sklearn.metrics.pairwise import euclidean_distances

        for target in self._predictor.targets:
            subset = df[df[target].notna()] if target in df.columns else df
            smiles_list = subset[smiles_col].dropna().tolist()
            training_scaled = self._scale_smiles_list(smiles_list)

            if len(training_scaled) == 0:
                continue

            D = euclidean_distances(training_scaled)
            np.fill_diagonal(D, np.inf)

            self._per_target[target] = {
                "training_scaled":   training_scaled,
                "baseline_nn_dists": D.min(axis=1),
                "n_training":        len(training_scaled),
            }

        return self

    def fit_from_csv(
        self,
        csv_path: str,
        smiles_col: str = "SMILES",
    ) -> "ApplicabilityDomain":
        """Read the dataset CSV and call fit().

        Parameters
        ----------
        csv_path : str
            Path to the dataset CSV (the same file used by surrogate.ipynb).
        smiles_col : str
            Name of the SMILES column (default: ``"SMILES"``).
        """
        df = pd.read_csv(csv_path)
        if smiles_col not in df.columns:
            raise ValueError(
                f"Column {smiles_col!r} not found in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        return self.fit(df, smiles_col=smiles_col)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, smiles: str) -> dict:
        """Compute per-target applicability-domain distances for one molecule.

        Parameters
        ----------
        smiles : str
            Query molecule as a SMILES string.

        Returns
        -------
        dict
            Maps each internal target name to a dict with:

            ``nn_distance``
                Euclidean distance to the nearest training molecule for this target.
            ``nn_distance_normalized``
                ``nn_distance / baseline_mean``.  ~1 = typical spacing;
                >> 1 = extrapolation.
            ``baseline_mean``
                Mean within-training NN distance for this target's training set.
            ``n_training``
                Number of molecules in this target's training set.
            ``in_domain``
                Heuristic: True when ``nn_distance <= baseline_mean``.
        """
        if not self._per_target:
            raise RuntimeError("Call fit() or fit_from_csv() before query().")

        X = self._predictor._build_features(smiles, self._feature_schema)
        X_arr = X.values.astype(float)
        nan_mask = np.isnan(X_arr)
        if nan_mask.any():
            X_arr[nan_mask] = self._fill_values[np.where(nan_mask)[1]]
        X_scaled = self._scaler.transform(X_arr)

        from sklearn.metrics.pairwise import euclidean_distances

        results = {}
        for target, data in self._per_target.items():
            dists = euclidean_distances(X_scaled, data["training_scaled"])[0]
            nn_dist = float(dists.min())
            baseline_mean = float(data["baseline_nn_dists"].mean())
            results[target] = {
                "nn_distance":            nn_dist,
                "nn_distance_normalized": nn_dist / baseline_mean if baseline_mean > 0 else float("inf"),
                "baseline_mean":          baseline_mean,
                "n_training":             data["n_training"],
                "in_domain":              nn_dist <= baseline_mean,
            }

        return results

    @property
    def baseline_stats(self) -> pd.DataFrame:
        """Per-target summary statistics of within-training NN distances.

        Returns a DataFrame indexed by target name with columns:
        n_training, mean, std, min, median, max.
        """
        if not self._per_target:
            raise RuntimeError("Call fit() or fit_from_csv() first.")
        rows = {}
        for target, data in self._per_target.items():
            d = data["baseline_nn_dists"]
            rows[target] = {
                "n_training": data["n_training"],
                "mean":       float(d.mean()),
                "std":        float(d.std()),
                "min":        float(d.min()),
                "median":     float(np.median(d)),
                "max":        float(d.max()),
            }
        return pd.DataFrame(rows).T


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _read_input_file(path: str) -> list:
    """Read (SMILES, name) pairs from a file.

    Each non-empty, non-comment line must start with a SMILES string.
    An optional name may follow, separated by a tab or a space:
        CCCCCCCC(=O)O
        CCCCCCCC(=O)O    octanoic acid
        CCCCCCCC(=O)O\toctanoic acid
    Lines starting with '#' are treated as comments.
    """
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smiles = parts[0]
            name = parts[1].strip() if len(parts) > 1 else None
            entries.append((smiles, name))
    return entries


def _print_result(result: dict, name: str, args) -> None:
    """Print the prediction output for one molecule."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(result["prediction"].to_string())

    if args.uncertainty:
        print("\nUncertainty (across outer test splits):")
        print(result["uncertainty"].to_string())

    if args.explain:
        print("\nTop SHAP features per target:")
        for label, data in result["explanation"].items():
            print(f"\n  {label}")
            for feat, val in data["top_features"].items():
                std = data["top_feature_uncertainty"][feat]
                print(f"    {feat:40s}  {val:.4f} ± {std:.4f}")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict surfactant properties from a SMILES string.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python predict.py "CCCCCCCCCC(=O)O"
  python predict.py "CCCCCCCCCC(=O)O" --uncertainty --explain
  python predict.py "CCCCCCCCCC(=O)O" --training-csv ../data/SurfPro-MD.csv
  python predict.py --input-file molecules.txt
  python predict.py --input-file molecules.txt --training-csv ../data/SurfPro-MD.csv
        """,
    )
    parser.add_argument(
        "smiles", nargs="?", default=None,
        help="SMILES string of the molecule to predict. Omit when using --input-file.",
    )
    parser.add_argument(
        "--input-file", "-i", default=None, metavar="PATH",
        help=(
            "Path to a file with one molecule per line. "
            "Each line: SMILES, or SMILES<tab>Name, or SMILES<space>Name. "
            "Lines starting with '#' are ignored. "
            "When provided, the positional smiles argument is not required."
        ),
    )
    parser.add_argument(
        "--model", default="models.pkl",
        help="Path to models.pkl produced by surrogate.ipynb (default: models.pkl).",
    )
    parser.add_argument(
        "--uncertainty", action="store_true",
        help="Report mean ± std across all 5 outer test splits.",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print SHAP feature importances (requires the shap package, may be slow).",
    )
    parser.add_argument(
        "--test-id", type=int, default=0,
        help="Outer test split to use for point predictions (0–4, default: 0).",
    )
    parser.add_argument(
        "--training-csv", default=None, metavar="PATH",
        help=(
            "Path to the training CSV (same file used by surrogate.ipynb). "
            "When provided, per-target applicability-domain distances are added "
            "as a column in the predictions table."
        ),
    )
    parser.add_argument(
        "--smiles-col", default="SMILES", metavar="COL",
        help="SMILES column name in the training CSV (default: SMILES).",
    )
    args = parser.parse_args()

    if args.smiles is None and args.input_file is None:
        parser.error("Provide either a SMILES string or --input-file.")

    predictor = SurfProPredictor(args.model)

    ad = None
    if args.training_csv:
        print("Building per-target applicability domain (this may take a moment)…")
        ad = ApplicabilityDomain(predictor)
        ad.fit_from_csv(args.training_csv, smiles_col=args.smiles_col)

    molecules = (
        _read_input_file(args.input_file)
        if args.input_file
        else [(args.smiles, None)]
    )

    for idx, (smiles, name) in enumerate(molecules, start=1):
        label = name if name is not None else f"molecule {idx}"
        try:
            result = predictor.predict(
                smiles,
                test_id=args.test_id,
                return_ci=args.uncertainty,
                return_explanation=args.explain,
                ad=ad,
            )
            _print_result(result, label, args)
        except Exception as exc:
            print(f"\n{'='*60}")
            print(f"  {label}  [ERROR]")
            print(f"{'='*60}")
            print(f"  {exc}")


if __name__ == "__main__":
    main()
