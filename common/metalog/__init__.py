"""
Metalog Distributions - Both Version 1.0 and 2.0

A flexible quantile-parameterized distribution family based on the metalog
distribution introduced by Keelin (2016).

This package provides BOTH implementations:

**Metalog 1.0** (Keelin 2016 - Original):
- Original basis function ordering
- Terms 1-6 identical to 2.0
- Terms 7, 8, 11, 12, etc. differ from 2.0
- May have rank deficiency issues at certain term counts

**Metalog 2.0** (Baucells et al. 2024 - Updated):
- Updated basis function ordering  
- Guaranteed full rank design matrix
- Better numerical stability
- Recommended for new applications

Supports unbounded, semibounded, and bounded domains.
"""

# Metalog 2.0 (Recommended - Updated ordering from "On the Properties")
from .metalog_v2 import (
    Metalog as Metalog_v2,
    LogMetalog as LogMetalog_v2,
    LogitMetalog as LogitMetalog_v2,
    MetalogError
)

# Default exports use version 2.0 (recommended)
Metalog = Metalog_v2
LogMetalog = LogMetalog_v2
LogitMetalog = LogitMetalog_v2


def __getattr__(name):
    """
    Lazily import Metalog 1.0 classes only when explicitly requested.

    This keeps import-time overhead low for cloud experiments that only use
    Metalog 2.0.
    """
    if name in {'Metalog_v1', 'LogMetalog_v1', 'LogitMetalog_v1'}:
        from .metalog_v1 import Metalog as _Metalog_v1
        from .metalog_v1 import LogMetalog as _LogMetalog_v1
        from .metalog_v1 import LogitMetalog as _LogitMetalog_v1
        mapping = {
            'Metalog_v1': _Metalog_v1,
            'LogMetalog_v1': _LogMetalog_v1,
            'LogitMetalog_v1': _LogitMetalog_v1,
        }
        return mapping[name]
    raise AttributeError(f"module 'metalog' has no attribute {name!r}")

__all__ = [
    # Version 2.0 (default, recommended)
    'Metalog',
    'LogMetalog',
    'LogitMetalog',
    'MetalogError',
    
    # Explicit version imports
    'Metalog_v1',
    'LogMetalog_v1',
    'LogitMetalog_v1',
    'Metalog_v2',
    'LogMetalog_v2',
    'LogitMetalog_v2',
]

__version__ = '2.0.0'
__version_v1__ = '1.0.0'
__version_v2__ = '2.0.0'
