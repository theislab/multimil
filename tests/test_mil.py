"""Smoke tests for MILClassifier: run on a small subset of real HLCA data.

Tests are skipped when tests/hlca_tutorial.h5ad is not present.
They cover:
  - classification: setup → train → get_model_output → load_query_data (query predict)
  - regression: same pipeline with a continuous target
"""

import os

import numpy as np
import pytest
import scanpy as sc

DATA_PATH = os.path.join(os.path.dirname(__file__), "hlca_tutorial.h5ad")

SAMPLE_KEY = "sample"
DISEASE_KEY = "disease"
AGE_KEY = "age_or_mean_of_age_range"

# Keep only this many samples per class for smoke tests — fast but real data
N_SAMPLES_PER_CLASS = 4
# Cap cells per sample to keep the test snappy
MAX_CELLS_PER_SAMPLE = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hlca_small():
    """Load HLCA and return a tiny balanced subset for classification tests."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Tutorial data not found at {DATA_PATH}. Skipping smoke tests.")

    adata = sc.read_h5ad(DATA_PATH)
    adata.obs[DISEASE_KEY] = adata.obs[DISEASE_KEY].astype("category")

    classes = adata.obs[DISEASE_KEY].cat.categories.tolist()
    samples_per_class = {}
    for cls in classes:
        cls_samples = adata.obs.loc[adata.obs[DISEASE_KEY] == cls, SAMPLE_KEY].unique()
        samples_per_class[cls] = list(cls_samples[:N_SAMPLES_PER_CLASS])

    selected_samples = [s for ss in samples_per_class.values() for s in ss]
    sub = adata[adata.obs[SAMPLE_KEY].isin(selected_samples)].copy()

    # Cap cells per sample
    keep_idx = []
    rng = np.random.default_rng(0)
    for sample in sub.obs[SAMPLE_KEY].unique():
        idx = np.where(sub.obs[SAMPLE_KEY] == sample)[0]
        chosen = rng.choice(idx, size=min(len(idx), MAX_CELLS_PER_SAMPLE), replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx.sort()
    sub = sub[keep_idx].copy()

    # Sort by sample (required by the model)
    sub = sub[sub.obs[SAMPLE_KEY].sort_values().index].copy()
    return sub


@pytest.fixture(scope="session")
def hlca_regression_small():
    """Tiny subset for regression tests (healthy samples with known age)."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Tutorial data not found at {DATA_PATH}. Skipping smoke tests.")

    adata = sc.read_h5ad(DATA_PATH)
    adata = adata[~adata.obs[AGE_KEY].isna()].copy()

    selected_samples = list(adata.obs[SAMPLE_KEY].unique()[:8])
    sub = adata[adata.obs[SAMPLE_KEY].isin(selected_samples)].copy()

    rng = np.random.default_rng(0)
    keep_idx = []
    for sample in sub.obs[SAMPLE_KEY].unique():
        idx = np.where(sub.obs[SAMPLE_KEY] == sample)[0]
        chosen = rng.choice(idx, size=min(len(idx), MAX_CELLS_PER_SAMPLE), replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx.sort()
    sub = sub[keep_idx].copy()

    sub = sub[sub.obs[SAMPLE_KEY].sort_values().index].copy()
    return sub


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_ref_query(adata, sample_key, n_query_samples=2):
    """Hold out a few samples as query, rest as reference."""
    samples = list(adata.obs[sample_key].unique())
    query_samples = samples[-n_query_samples:]
    ref_samples = samples[:-n_query_samples]
    ref = adata[adata.obs[sample_key].isin(ref_samples)].copy()
    query = adata[adata.obs[sample_key].isin(query_samples)].copy()
    return ref, query


# ---------------------------------------------------------------------------
# Tests — package sanity
# ---------------------------------------------------------------------------


def test_package_has_version():
    import multimil

    assert multimil.__version__ is not None


# ---------------------------------------------------------------------------
# Tests — classification
# ---------------------------------------------------------------------------


def test_classification_smoke(hlca_small):
    import multimil as mtm

    ref, query = _split_ref_query(hlca_small, SAMPLE_KEY, n_query_samples=2)

    classification_keys = [DISEASE_KEY]
    categorical_covariate_keys = classification_keys + [SAMPLE_KEY]

    mtm.model.MILClassifier.setup_anndata(
        ref,
        categorical_covariate_keys=categorical_covariate_keys,
    )

    mil = mtm.model.MILClassifier(
        ref,
        classification=classification_keys,
        sample_key=SAMPLE_KEY,
        class_loss_coef=0.1,
    )

    mil.train(max_epochs=2, save_best=False)

    assert mil.is_trained_

    mil.get_model_output()

    # Cell-level outputs
    assert "cell_attn" in ref.obs.columns, "cell_attn missing from obs"
    assert "bags" in ref.obs.columns, "bags missing from obs"
    assert f"predicted_{DISEASE_KEY}" in ref.obs.columns, f"predicted_{DISEASE_KEY} missing from obs"
    assert f"full_predictions_{DISEASE_KEY}" in ref.obsm, f"full_predictions_{DISEASE_KEY} missing from obsm"
    assert ref.obsm[f"full_predictions_{DISEASE_KEY}"].shape == (
        ref.n_obs,
        len(ref.obs[DISEASE_KEY].cat.categories),
    ), "full_predictions shape mismatch"

    # Bag-level outputs
    assert f"bag_true_{DISEASE_KEY}" in ref.uns, "bag_true missing from uns"
    assert f"bag_full_predictions_{DISEASE_KEY}" in ref.uns, "bag_full_predictions missing from uns"

    # Attention is non-negative and finite
    assert np.all(np.isfinite(ref.obs["cell_attn"].values)), "cell_attn contains non-finite values"
    assert np.all(ref.obs["cell_attn"].values >= 0), "cell_attn contains negative values"


def test_classification_load_query_data(hlca_small):
    """Train on ref, transfer to query via load_query_data — the 'new data' prediction path."""
    import multimil as mtm

    ref, query = _split_ref_query(hlca_small, SAMPLE_KEY, n_query_samples=2)

    classification_keys = [DISEASE_KEY]
    categorical_covariate_keys = classification_keys + [SAMPLE_KEY]

    mtm.model.MILClassifier.setup_anndata(
        ref,
        categorical_covariate_keys=categorical_covariate_keys,
    )

    mil = mtm.model.MILClassifier(
        ref,
        classification=classification_keys,
        sample_key=SAMPLE_KEY,
        class_loss_coef=0.1,
    )
    mil.train(max_epochs=2, save_best=False)

    # Transfer weights to query — no re-training
    query_model = mtm.model.MILClassifier.load_query_data(query, mil)
    assert query_model.is_trained_, "load_query_data model should be marked as trained"

    query_model.get_model_output()

    assert "cell_attn" in query.obs.columns, "cell_attn missing from query obs"
    assert f"predicted_{DISEASE_KEY}" in query.obs.columns, f"predicted_{DISEASE_KEY} missing from query obs"
    assert f"full_predictions_{DISEASE_KEY}" in query.obsm, f"full_predictions_{DISEASE_KEY} missing from query obsm"
    assert query.obsm[f"full_predictions_{DISEASE_KEY}"].shape[0] == query.n_obs


# ---------------------------------------------------------------------------
# Tests — regression
# ---------------------------------------------------------------------------


def test_regression_smoke(hlca_regression_small):
    import multimil as mtm

    ref, query = _split_ref_query(hlca_regression_small, SAMPLE_KEY, n_query_samples=1)

    regression_keys = [AGE_KEY]
    categorical_covariate_keys = [SAMPLE_KEY]
    continuous_covariate_keys = regression_keys

    mtm.model.MILClassifier.setup_anndata(
        ref,
        categorical_covariate_keys=categorical_covariate_keys,
        continuous_covariate_keys=continuous_covariate_keys,
    )

    mil = mtm.model.MILClassifier(
        ref,
        regression=regression_keys,
        sample_key=SAMPLE_KEY,
    )
    mil.train(max_epochs=2, save_best=False)

    assert mil.is_trained_

    mil.get_model_output()

    assert "cell_attn" in ref.obs.columns
    assert f"predicted_{AGE_KEY}" in ref.obs.columns
    assert f"full_predictions_{AGE_KEY}" in ref.obsm
    assert f"bag_true_{AGE_KEY}" in ref.uns
    assert f"bag_full_predictions_{AGE_KEY}" in ref.uns

    # Transfer to query
    query_model = mtm.model.MILClassifier.load_query_data(query, mil)
    query_model.get_model_output()
    assert f"predicted_{AGE_KEY}" in query.obs.columns
