from importlib.metadata import version

from . import data, dataloaders, distributions, model, module, nn, utils

__all__ = ["data", "dataloaders", "distributions", "model", "module", "nn", "utils"]

__version__ = version("multimil")
