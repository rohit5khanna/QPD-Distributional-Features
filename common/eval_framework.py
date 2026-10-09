"""
The paper's evaluation framework -- ONE definition of the grid, validity,
density, modes and W1, imported by every section script and every figure.

    1. GRID      p = 0.001, 0.002, ..., 0.999 (999 points). Nothing is
                 computed, extrapolated or drawn outside it.
    2. VALIDITY  Q strictly increasing along the grid (delta-p monotonicity,
                 delta = 0.001). Nothing else (no check on a library pdf).
    3. DENSITY   from the quantile values only, by differences between
                 neighbouring grid points:
                     f(p_i) = (p_{i+1} - p_{i-1}) / (Q(p_{i+1}) - Q(p_{i-1}))
                 one-sided at p = 0.001 and 0.999; drawn at x = Q(p_i).
    4. MODES     interior only: a local maximum of the density on the grid
                 (higher than both neighbours; scipy.signal.find_peaks) whose
                 prominence is at least 1% of ITS OWN height (relative
                 prominence -- a local criterion, unaffected by the density
                 anywhere else). Prominence = how far the peak rises above the
                 lowest point reached, on either side, before a higher point
                 or the end of the grid (the higher of the two lows). The
                 first and last grid points are never modes. A fit with no
                 peak is unimodal.
    5. W1        Equation 6: mean of |Q_fit(p) - EQF(p)| over the grid
                 points INSIDE THE DATA RANGE, 1/(N+1) <= p <= N/(N+1),
                 divided by EQF(0.9) - EQF(0.1). EQF = the reference sample's
                 order statistics at p_i = i/(N+1), linearly interpolated.
                 Grid points beyond the data range (p < 1/(N+1) or
                 p > N/(N+1)) are not used: the data carry no observation
                 there. (eqf() still returns values on the whole grid, held at
                 the smallest / largest observation, for drawing only.)
                 w1_fullgrid() keeps the previous whole-grid definition for
                 comparison.

NUMERICS (no change to the definitions). For Log / Logit models the quantile
is Q(p) = L + exp(z(p)) or L + (U-L) expit(z(p)), z the fitted QPD on the
unbounded scale. Validity is checked on z (the transform is strictly
increasing, so "Q increasing" and "z increasing" are the same statement), and
the differences Q(p_{i+1}) - Q(p_{i-1}) are formed from z without
cancellation. This matters only where Q is within rounding of a bound, where
differencing Q itself returns 0 (e.g. x = 1 - 1e-17 for Logit fits).
"""
import numpy as np
from scipy.signal import find_peaks

GRID = np.round(np.arange(1, 1000) * 1e-3, 3)        # 0.001 .. 0.999
DELTA = 0.001
RELATIVE_PROMINENCE = 0.01          # of the peak's own height


# ---------------------------------------------------------------- the model
def _parts(m):
    """(z on GRID, transform, lb, ub) for a fitted QPD object."""
    name = type(m).__name__
    inner = getattr(m, 'metalog', None) or getattr(m, 'qflex', None)
    if name.startswith('Logit'):
        kind, lb, ub = 'logit', float(m.lower_bound), float(m.upper_bound)
    elif name.startswith('Log'):
        kind, lb, ub = 'log', float(m.lower_bound), np.inf
    else:
        kind, lb, ub, inner = 'none', -np.inf, np.inf, m
    with np.errstate(all='ignore'):
        z = np.asarray(inner.quantile(GRID), float)
    return z, kind, lb, ub


def _logcosh(a):
    a = np.abs(a)
    return a + np.log1p(np.exp(-2.0 * a)) - np.log(2.0)


def _log_dq(z1, z2, kind, lb, ub):
    """log(Q(z2) - Q(z1)) for z2 > z1, without cancellation."""
    with np.errstate(all='ignore'):
        d = z2 - z1
        if kind == 'none':
            return np.log(d)
        if kind == 'log':                           # e^z2 - e^z1
            return z2 + np.log(-np.expm1(-d))
        # (U-L)(expit z2 - expit z1) = (U-L) sinh(d/2) / (2 cosh(z1/2) cosh(z2/2))
        return (np.log(ub - lb) + d / 2 + np.log(-np.expm1(-d)) - 2 * np.log(2.0)
                - _logcosh(z1 / 2) - _logcosh(z2 / 2))


def quantile(m):
    """Q(p) on GRID, on the data scale."""
    return np.asarray(m.quantile(GRID), float)


def is_valid(m):
    """Definition 2. False for a missing fit."""
    if m is None:
        return False
    z = _parts(m)[0]
    return bool(np.all(np.isfinite(z)) and np.all(np.diff(z) > 0))


def density(m):
    """(Q, f) on GRID for a VALID fit (Definition 3). f > 0 everywhere."""
    z, kind, lb, ub = _parts(m)
    lf = np.empty_like(GRID)
    lf[1:-1] = np.log(GRID[2:] - GRID[:-2]) - _log_dq(z[:-2], z[2:], kind, lb, ub)
    lf[0] = np.log(GRID[1] - GRID[0]) - _log_dq(z[0], z[1], kind, lb, ub)
    lf[-1] = np.log(GRID[-1] - GRID[-2]) - _log_dq(z[-2], z[-1], kind, lb, ub)
    with np.errstate(over='ignore'):
        f = np.exp(np.minimum(lf, np.log(1e300)))
    return quantile(m), f


def density_from_quantile(q):
    """Definition 3 for a quantile already on GRID (e.g. a reference model
    such as GEV, or a true distribution). Plain differences of q."""
    q = np.asarray(q, float)
    f = np.empty_like(q)
    with np.errstate(divide='ignore', invalid='ignore'):
        f[1:-1] = (GRID[2:] - GRID[:-2]) / (q[2:] - q[:-2])
        f[0] = (GRID[1] - GRID[0]) / (q[1] - q[0])
        f[-1] = (GRID[-1] - GRID[-2]) / (q[-1] - q[-2])
    return f


def modes_from_density(q, f):
    """Definition 4 on given (Q, f) on GRID.
    Returns (n_peaks, locations, heights) -- the contract of
    mode_utils.detect_modes_from_arrays: n_peaks = 0 when the density has no
    interior peak (count it as 1 mode), -1 when f is unusable."""
    q, f = np.asarray(q, float), np.asarray(f, float)
    if f.shape != GRID.shape or not np.all(np.isfinite(f)) or not np.all(f > 0):
        return -1, None, None
    pk, props = find_peaks(f, prominence=0.0)          # every local maximum
    keep = props['prominences'] >= RELATIVE_PROMINENCE * f[pk]
    pk = pk[keep]
    return int(len(pk)), q[pk], f[pk]


def modes(m):
    """Definition 4 for a fitted QPD. (n_peaks, locations, heights); -1 if invalid."""
    if not is_valid(m):
        return -1, None, None
    return modes_from_density(*density(m))


def n_modes(n_peaks):
    """Mode count as reported: a density with no interior peak is unimodal."""
    return max(1, int(n_peaks)) if n_peaks is not None and n_peaks >= 0 else None


# ---------------------------------------------------------------------- W1
def plotting_positions(n):
    return np.arange(1, n + 1) / (n + 1)


def eqf(x_sorted):
    """Definition 5: interpolated EQF on GRID, held flat outside the data."""
    x_sorted = np.sort(np.asarray(x_sorted, float))
    return np.interp(GRID, plotting_positions(len(x_sorted)), x_sorted)


def interdecile(x_sorted):
    x_sorted = np.sort(np.asarray(x_sorted, float))
    pp = plotting_positions(len(x_sorted))
    return float(np.interp(0.9, pp, x_sorted) - np.interp(0.1, pp, x_sorted))


def data_mask(n):
    """Grid points inside the data range [1/(n+1), n/(n+1)] of a sample of size n."""
    pp = plotting_positions(int(n))
    return (GRID >= pp[0] - 1e-12) & (GRID <= pp[-1] + 1e-12)


def w1_on_grid(q_on_grid, eqf_on_grid, n, divisor=None):
    """Equation 6 from arrays on GRID: mean |Q - EQF| over data_mask(n).
    Returns (raw, raw / divisor) -- NaN normalized if no divisor."""
    mk = data_mask(n)
    d = np.abs(np.asarray(q_on_grid, float) - np.asarray(eqf_on_grid, float))
    raw = float(np.mean(d[mk]))
    return raw, (raw / divisor if divisor else np.nan)


def w1(q_on_grid, x_ref):
    """Equation 6 against the reference sample x_ref, over its data range.
    Returns (raw, normalized): raw in data units, normalized / interdecile."""
    return w1_on_grid(q_on_grid, eqf(x_ref), len(x_ref), interdecile(x_ref))


def w1_fullgrid(q_on_grid, x_ref):
    """Previous definition (whole grid, EQF held flat beyond the data). Comparison only."""
    raw = float(np.mean(np.abs(np.asarray(q_on_grid, float) - eqf(x_ref))))
    return raw, raw / interdecile(x_ref)


# ------------------------------------------------------------- self-test
if __name__ == '__main__':
    import sys, os
    HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
    from metalog.metalog_v2 import Metalog, LogMetalog, LogitMetalog
    from qflex.core import QFlex
    from qflex.transforms import LogQFlex, LogitQFlex
    from qflex.constraints import ConstraintType as CT
    assert len(GRID) == 999 and GRID[0] == 0.001 and GRID[-1] == 0.999
    rng = np.random.default_rng(1)
    # 1. density of a known distribution: Exp(1) on the log scale
    q = -np.log1p(-GRID); f = density_from_quantile(q)
    true = 1 - GRID                                  # f(Q(p)) = 1 - p
    mid = (GRID >= 0.05) & (GRID <= 0.95)            # difference error ~ h^2/(1-p)^2 near the ends
    assert np.max(np.abs(f[mid] / true[mid] - 1)) < 1e-3, 'central difference'
    # 2. every transform: density from z equals plain differencing of Q
    x01 = np.sort(rng.beta(2, 5, 300)); xpos = np.sort(rng.lognormal(1, .5, 300))
    xun = np.sort(rng.normal(0, 1, 300)); pp = plotting_positions(300)
    fits = [Metalog(xun, pp, 5), QFlex(xun, pp, 7, CT.NONE),
            LogMetalog(xpos, pp, 0, 5), LogQFlex(xpos, pp, 0, 7, CT.A),
            LogitMetalog(x01, pp, 0, 1, 5), LogitQFlex(x01, pp, 0, 1, 7, CT.TA)]
    for m in fits:
        assert is_valid(m), type(m).__name__
        qg, fg = density(m)
        assert np.max(np.abs(fg / density_from_quantile(qg) - 1)) < 1e-6, type(m).__name__
        z, kind, lb, ub = _parts(m)
        back = z if kind == 'none' else (lb + np.exp(z) if kind == 'log' else lb + (ub - lb) / (1 + np.exp(-z)))
        assert np.allclose(back, qg, rtol=1e-12, atol=1e-12), 'transform ' + type(m).__name__
        n, _l, _h = modes(m); assert n >= 0
    # 3. a non-monotone quantile is invalid
    class Fake:
        def quantile(self, p): return np.sin(6 * np.asarray(p))
    assert not is_valid(Fake())
    # 4. modes: a two-bump density is found as two peaks, a monotone one as none
    qq = np.interp(GRID, np.linspace(0, 1, 5), [0, 1, 1.2, 1.4, 3.0])   # piecewise-linear Q
    n, _, _ = modes_from_density(qq + 1e-9 * GRID, density_from_quantile(qq + 1e-9 * GRID))
    assert n >= 0
    n, _, _ = modes_from_density(-np.log1p(-GRID), density_from_quantile(-np.log1p(-GRID)))
    assert n == 0 and n_modes(n) == 1
    # 4b. relative prominence: a huge spike at the grid edge does not hide an
    #     interior bump, and a 0.5%-of-height ripple is not a mode
    xx = np.linspace(0, 1, len(GRID))
    f_test = np.exp(-((xx - 0.3) / 0.05) ** 2) + 0.5 * np.exp(-((xx - 0.6) / 0.05) ** 2) + 0.05 + 1e6 * (xx > 0.995)
    n, _, _ = modes_from_density(xx, f_test); assert n == 2, n
    f_rip = 1.0 + 0.002 * np.sin(200 * xx) * (xx > 0.5) - (xx - 0.3) ** 2
    n, _, _ = modes_from_density(xx, f_rip); assert n == 1, n
    # 5. W1: EQF reproduces the data at the plotting positions; W1 of the
    #    EQF against itself is 0
    xs = np.sort(rng.normal(size=999)); assert np.allclose(eqf(xs), xs)
    assert w1(eqf(xs), xs)[0] == 0.0
    # 5b. W1 uses only the data range: a fit that differs from the EQF only
    #     beyond the data has W1 = 0; the whole-grid version does not
    small5 = np.sort(rng.normal(size=50)); mk = data_mask(50)
    assert mk.sum() == int(np.sum((GRID >= 1/51) & (GRID <= 50/51)))
    qq5 = eqf(small5).copy(); qq5[~mk] += 10.0
    assert w1(qq5, small5)[0] == 0.0 and w1_fullgrid(qq5, small5)[0] > 0
    small = np.array([2.0, 5.0, 9.0])               # flat outside [1/4, 3/4]
    e = eqf(small); assert e[0] == 2.0 and e[-1] == 9.0 and abs(e[374] - 3.5) < 1e-12
    print('eval_framework self-test OK')
