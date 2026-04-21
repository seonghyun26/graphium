import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# Suppress RDKit C++ warnings (e.g. "not removing hydrogen atom without neighbors")
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")

from ._version import __version__

from .config import load_config

from . import utils
from . import features
from . import data
from . import nn
from . import trainer
