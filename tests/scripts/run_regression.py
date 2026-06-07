"""Tutorial-parity script: regression (age prediction on HLCA subset).

Mirrors the mil_regression.ipynb notebook workflow: KFold cross-validation,
train on reference split, predict on query split via load_query_data.

Usage
-----
uv run python tests/scripts/run_regression.py
uv run python tests/scripts/run_regression.py --max-epochs 20 --n-splits 3
uv run python tests/scripts/run_regression.py --subset-samples 20  # faster CPU run
"""

import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

import multimil as mtm


def parse_args():
    p = argparse.ArgumentParser(description="MultiMIL regression tutorial parity script")
    p.add_argument("--data", default="tests/hlca_tutorial.h5ad", help="Path to HLCA h5ad")
    p.add_argument("--max-epochs", type=int, default=20, help="Training epochs")
    p.add_argument("--n-splits", type=int, default=3, help="KFold splits")
    p.add_argument(
        "--subset-samples",
        type=int,
        default=None,
        help="Use only N samples total (speeds up CPU runs; None = use all)",
    )
    p.add_argument(
        "--max-cells-per-sample",
        type=int,
        default=None,
        help="Cap cells per sample (None = use all)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    scvi.settings.seed = 0
    print(f"scvi-tools version: {scvi.__version__}")

    print(f"Loading data from {args.data} ...")
    adata = sc.read_h5ad(args.data)

    sample_key = "sample"
    condition_key = "age_or_mean_of_age_range"
    regression_keys = [condition_key]
    categorical_covariate_keys = [sample_key]
    continuous_covariate_keys = regression_keys

    # Regression tutorial uses only healthy samples with known age
    adata = adata[~adata.obs[condition_key].isna()].copy()
    print(f"After age filter: {adata.n_obs} cells")

    if args.subset_samples is not None:
        rng = np.random.default_rng(0)
        all_samples = adata.obs[sample_key].unique()
        chosen = rng.choice(all_samples, size=min(len(all_samples), args.subset_samples), replace=False)
        adata = adata[adata.obs[sample_key].isin(chosen)].copy()
        print(f"Subset to {len(chosen)} samples.")

    if args.max_cells_per_sample is not None:
        rng = np.random.default_rng(0)
        keep_idx = []
        for sample in adata.obs[sample_key].unique():
            idx = np.where(adata.obs[sample_key] == sample)[0]
            chosen = rng.choice(idx, size=min(len(idx), args.max_cells_per_sample), replace=False)
            keep_idx.extend(chosen.tolist())
        adata = adata[sorted(keep_idx)].copy()
        print(f"Capped to {args.max_cells_per_sample} cells per sample.")

    print(f"Dataset: {adata.n_obs} cells, {adata.n_vars} features, {adata.obs[sample_key].nunique()} samples")

    samples = np.array(adata.obs[sample_key].unique())
    n_splits = args.n_splits
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    for i, (train_index, val_index) in enumerate(kf.split(samples)):
        train_samples = samples[train_index]
        val_samples = samples[val_index]
        adata.obs.loc[adata.obs[sample_key].isin(train_samples), f"split{i}"] = "train"
        adata.obs.loc[adata.obs[sample_key].isin(val_samples), f"split{i}"] = "val"

    for i in range(n_splits):
        print(f"\n{'='*60}")
        print(f"Split {i} / {n_splits - 1}")
        print(f"{'='*60}")

        query = adata[adata.obs[f"split{i}"] == "val"].copy()
        ref = adata[adata.obs[f"split{i}"] == "train"].copy()

        # Sort by sample (required)
        ref = ref[ref.obs[sample_key].sort_values().index].copy()
        query = query[query.obs[sample_key].sort_values().index].copy()

        print(f"Ref: {ref.n_obs} cells, {ref.obs[sample_key].nunique()} samples")
        print(f"Query: {query.n_obs} cells, {query.obs[sample_key].nunique()} samples")

        mtm.model.MILClassifier.setup_anndata(
            ref,
            categorical_covariate_keys=categorical_covariate_keys,
            continuous_covariate_keys=continuous_covariate_keys,
        )

        mil = mtm.model.MILClassifier(
            ref,
            regression=regression_keys,
            sample_key=sample_key,
        )

        mil.train(max_epochs=args.max_epochs)
        mil.get_model_output()

        # Predict on held-out query samples
        new_model = mtm.model.MILClassifier.load_query_data(query, mil)
        new_model.get_model_output()

        # Save cell attention back to full adata
        cell_attn_i = pd.concat([ref.obs["cell_attn"], query.obs["cell_attn"]])
        cell_attn_i.name = f"cell_attn_{i}"
        adata.obs = adata.obs.join(cell_attn_i, how="left")

        r2 = r2_score(query.obs[condition_key], query.obs[f"predicted_{condition_key}"])
        print(f"\nQuery R² (split {i}): {r2:.4f}")

    # Average attention across splits
    attn_cols = [f"cell_attn_{i}" for i in range(n_splits)]
    adata.obs["cell_attn"] = np.nanmean(adata.obs[attn_cols].values, axis=1)

    # Score top cells (top 10% by attention)
    mtm.utils.score_top_cells(adata)

    # Sample representations
    sample_reps = mtm.utils.get_sample_representations(adata, sample_key=sample_key, covs_to_keep=[condition_key])
    print(f"\nSample representations: {sample_reps}")

    print("\nDone.")


if __name__ == "__main__":
    main()
