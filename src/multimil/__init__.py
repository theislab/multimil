from importlib.metadata import version

from . import dataloaders, model, module, nn, utils

__all__ = ["dataloaders", "model", "module", "nn", "utils"]

__version__ = version("multimil")
