# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [1.0.0] - 2026-06-07

### Changed

- Migrated to scvi-tools ≥1.4 (Lightning 2.x): replaced `use_gpu` with `accelerator`/`devices`, updated `SaveBestState` → `SaveCheckpoint`, `_initialize_model`, and `_get_loaded_data` call sites.
- `GroupDataSplitter` now splits at the **sample level** (not cell level) using `external_indexing`, ensuring every validation bag is complete.
- Bumped `requires-python` to `>=3.12`; switched package manager to [uv](https://docs.astral.sh/uv/) with a committed `uv.lock`.
- numpy ≥2.0 and anndata ≥0.12 compatibility fixes (dtype identity check, pandas `groupby(observed=True)`).

### Fixed

- Single-layer regressor head used `z_dim` instead of `class_input_dim` when sample covariates are present.
- `.squeeze()` on 1-element tensors in `create_df` produced 0-d scalars; wrapped with `np.atleast_1d`.
- Fixed floating-point undercount in `GroupDataSplitter` (`floor(1.0 - 0.9) = 0` edge case).

## [0.3.2] - 2025-12-16

### Added

- tutorial for regression
- support for sample-covariate embeddings, experimental, only one-hot

### Changed

- removed `muon` from dependencies for tutorials
- changed the classification tutorial to include training across 3 CV folds

### Fixed

- fixed a bug in regression which was caused by a wrong key when accessing the continuous covariates registered with scvi-tools

## [0.3.1] - 2025-07-14

### Fixed

- Fixed a bug in `score_top_cells` that didn't set the specified `key_added` column in .obs to True.
- Fixed a bug in `score_top_cells` that set the `key_added` to be categorical, which resulted in wrong indexing in `get_sample_representations`.

## [0.3.0] - 2025-07-13

### Added

- **Utility functions**: Added `score_top_cells` and `get_sample_representations` to utils module
- `score_top_cells`: Function to identify and score top cells based on attention weights
- `get_sample_representations`: Function to aggregate cell-level data to sample-level representations

### Changed

- **Major refactoring**: Removed MultiVAE and MultiVAE_MIL models, keeping only MIL classifier
- **Code cleanup**: Removed MLP attention weight learning from Aggregator class
- **Parameter consistency**: Fixed default values between model and module classes
- **Dynamic z_dim**: Automatically infer z_dim from input data shape instead of hardcoded value

### Fixed

- **Make categorical covariates categorical**: Ensure the correct type in `setup_anndata`.
- **Improved error handling**: If the prediction covariate hasn't been registered with `setup_anndata`, throw an error.
- **Dead links to API and changelog**: Fixed in README.
