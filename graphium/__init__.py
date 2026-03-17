import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from ._version import __version__

from .config import load_config

from . import utils
from . import features
from . import data
from . import nn
from . import trainer
