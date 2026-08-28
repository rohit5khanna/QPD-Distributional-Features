"""
Johnson Power-Series Expansions (JPSE)

Implementation of the Johnson distribution system using quantile functions
as described in:

Bickel, J. E. (2026). Quantile-based power-series expansions of the Johnson
distribution system. Communications in Statistics - Theory and Methods.
DOI: 10.1080/03610926.2025.2612230
"""

from .johnson import JohnsonSU, JohnsonSL, JohnsonSB

__all__ = ['JohnsonSU', 'JohnsonSL', 'JohnsonSB']
