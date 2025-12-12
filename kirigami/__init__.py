"""
Kirigami data-generation core package.

This package namespaces the original `Structure.py` and `Utils.py` modules as:
- `kirigami.structure`
- `kirigami.utils`
"""

from .structure import GenericStructure, MatrixStructure

__all__ = ["GenericStructure", "MatrixStructure"]

