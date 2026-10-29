"""scELF: A unified statistical framework for single-cell eQTL fine-mapping."""

from .model import ScELF
from .backend import backend_info
from .coloc import ColocPIPResult, coloc_pip

__all__ = ["ScELF", "backend_info", "ColocPIPResult", "coloc_pip"]
__version__ = "0.1.0"
