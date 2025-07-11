# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

-   **Utility functions**: Added `score_top_cells` and `get_sample_representations` to utils module
  - `score_top_cells`: Function to identify and score top cells based on attention weights
  - `get_sample_representations`: Function to aggregate cell-level data to sample-level representations

### Changed

-   **Major refactoring**: Removed MultiVAE and MultiVAE_MIL models, keeping only MIL classifier
-   **Code cleanup**: Removed MLP attention weight learning from Aggregator class
-   **Parameter consistency**: Fixed default values between model and module classes
-   **Dynamic z_dim**: Automatically infer z_dim from input data shape instead of hardcoded value
