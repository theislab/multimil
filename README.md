# Multimodal weakly supervised learning to identify disease-specific changes in single-cell atlases

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/theislab/multimil/test.yaml?branch=main
[link-tests]: https://github.com/theislab/multimil/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/multimil
[badge-colab]: https://colab.research.google.com/assets/colab-badge.svg

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

- [API documentation][link-api]

and the tutorials:

- [Classification with MIL](https://multimil.readthedocs.io/en/latest/notebooks/mil_classification.html) [![Open In Colab][badge-colab]](https://colab.research.google.com/github/theislab/multimil/blob/main/docs/notebooks/mil_classification.ipynb)

## Installation

You need to have Python 3.10 or newer installed on your system. We recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

To create and activate a new environment:

```bash
mamba create --name multimil python=3.10
mamba activate multimil
```

Next, there are several alternative options to install multimil:

1. Install the latest release of `multimil` from [PyPI][link-pypi]:

```bash
pip install multimil
```

2. Or install the latest development version:

```bash
pip install git+https://github.com/theislab/multimil.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> **Multimodal Weakly Supervised Learning to Identify Disease-Specific Changes in Single-Cell Atlases**
>
> Anastasia Litinetskaya, Maiia Shulman, Soroor Hediyeh-zadeh, Amir Ali Moinfar, Fabiola Curion, Artur Szalata, Alireza Omidi, Mohammad Lotfollahi, and Fabian J. Theis. 2024. bioRxiv. https://doi.org/10.1101/2024.07.29.605625.

## Reproducibility

Code and notebooks to reproduce the results from the paper are available at [theislab/multimil_reproducibility](/https://github.com/theislab/multimil_reproducibility).

[issue-tracker]: https://github.com/theislab/multimil/issues
[changelog]: https://multimil.readthedocs.io/en/latest/changelog.html
[link-docs]: https://multimil.readthedocs.io
[link-api]: https://multimil.readthedocs.io/en/latest/api.html
[link-pypi]: https://pypi.org/project/multimil
