# Weakly supervised learning uncovers phenotypic signatures in single-cell data

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/multimil/test.yaml?branch=main
[link-tests]: https://github.com/theislab/multimil/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/multimil
[badge-colab]: https://colab.research.google.com/assets/colab-badge.svg

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api]

and the tutorials:

-   [Classification with MultiMIL](https://multimil.readthedocs.io/en/latest/notebooks/mil_classification.html) [![Open In Colab][badge-colab]](https://colab.research.google.com/github/theislab/multimil/blob/main/docs/notebooks/mil_classification.ipynb)
-   [Regression with MultiMIL](https://multimil.readthedocs.io/en/latest/notebooks/mil_regression.html) [![Open In Colab][badge-colab]](https://colab.research.google.com/github/theislab/multimil/blob/main/docs/notebooks/mil_regression.ipynb)

Please also check out our [sample prediction pipeline](https://github.com/theislab/sample-prediction-pipeline), which contains MultiMIL and several other baselines.

## Installation

You need to have Python 3.12 or newer installed on your system. We recommend using [uv](https://docs.astral.sh/uv/) for environment management.

Install the latest release of `multimil` from [PyPI][link-pypi]:

```bash
uv pip install multimil
```

Or install the latest development version:

```bash
uv pip install git+https://github.com/theislab/multimil.git@main
```

Alternatively, with plain pip:

```bash
pip install multimil
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

Weakly supervised learning uncovers phenotypic signatures in single-cell data

Anastasia Litinetskaya, Soroor Hediyeh-zadeh, Amir Ali Moinfar, Mohammad Lotfollahi, Fabian J. Theis

bioRxiv 2024.07.29.605625; doi: https://doi.org/10.1101/2024.07.29.605625

## Reproducibility

Code and notebooks to reproduce the results from the paper are available at [theislab/multimil_reproducibility](https://github.com/theislab/multimil_reproducibility).

[issue-tracker]: https://github.com/theislab/multimil/issues
[changelog]: https://multimil.readthedocs.io/en/latest/changelog.html
[link-docs]: https://multimil.readthedocs.io
[link-api]: https://multimil.readthedocs.io/en/latest/api.html
[link-pypi]: https://pypi.org/project/multimil
