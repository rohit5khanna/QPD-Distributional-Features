"""
QFlex Distributions

A flexible quantile-parameterized distribution family with support for
unbounded, semibounded, and bounded domains, plus optional constraints
to ensure valid probability densities.
"""

from .core import QFlex
from .constraints import ConstraintType, QFlexError
from .mono_verification import check_proposition4, check_delta_p_monotonicity


def __getattr__(name):
    """
    Lazily import LogQFlex/LogitQFlex only when explicitly requested.

    marimo's WASM export bundles local packages by statically tracing the
    import graph starting from app.py's own import statements; it doesn't
    parse this __init__.py's imports as part of that trace. `transforms.py`
    is the only file in this package reachable *exclusively* through here
    (every other submodule is also imported, directly or transitively, from
    core.py), so it can end up left out of the exported wheel even though
    this __init__.py still runs and would otherwise import it eagerly. Since
    the notebook doesn't use LogQFlex/LogitQFlex, keeping this import lazy
    avoids depending on transforms.py being bundled at all.
    """
    if name in {'LogQFlex', 'LogitQFlex'}:
        from .transforms import LogQFlex as _LogQFlex, LogitQFlex as _LogitQFlex
        mapping = {'LogQFlex': _LogQFlex, 'LogitQFlex': _LogitQFlex}
        return mapping[name]
    raise AttributeError(f"module 'qflex' has no attribute {name!r}")


__all__ = [
    'QFlex',
    'LogQFlex',
    'LogitQFlex',
    'QFlexError',
    'ConstraintType',
    'check_proposition4',
    'check_delta_p_monotonicity',
]

__version__ = '1.0.0'
