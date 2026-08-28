"""
Shared modality-test implementations used across this repo's experiments.

IMPORTANT: this file holds TWO distinct, non-interchangeable Silverman
critical-bandwidth test implementations, kept separate on purpose. They were
developed independently for different experiments and are NOT numerically
equivalent to each other (different bisection schedules and different
null-bootstrap resampling schemes) -- do not substitute one for the other
without re-validating against the numbers already reported in the paper.

  1. MC-SYNTHETIC FAMILY  (apply_silverman_test, apply_hartigan_dip_test)
     Used by: MC_Cloud/mc_jpse_comprehensive.py,
              MC_Cloud_Bimodal/mc_bimodal_comprehensive.py,
              MC_Cloud_Empirical/bootstrap_empirical.py
     Silverman: bisects h between [0.01*h0, 5*h0] (h0 = 1.06*std*n^-0.2),
     counts modes via a manual Gaussian-KDE grid + scipy.signal.find_peaks
     (prominence=1e-4), and calibrates the null via Hall & York (2001)
     resampling from gaussian_kde(x, bw=h_null/std).

  2. FISH-JITTER FAMILY  (jitter_find_h_crit, jitter_silverman_pvalue)
     Used by: Fish_Modality_Tests/fish_silverman_jitter_sweep.py,
              Fish_Modality_Tests/fish_silverman_jitter_sweep_cloud.py
     Silverman: bisects h starting from h_max = 0.9*std*n^-0.2 (expand x2,
     contract /2), counts modes via a manual Gaussian-KDE grid +
     scipy.signal.find_peaks (prominence = 5% of peak PDF height), and
     calibrates the null via the explicit Hall & York (2001)
     variance-preserving bootstrap formula
         x* = mu + (X_boot - mu + h_crit * Z) / sqrt(1 + h_crit^2 / s^2).

  Fish_Modality_Tests/fish_kde_bandwidth_robustness.py has its OWN third
  find_h_crit/silverman_pvalue variant (different bisection constants) that
  is intentionally NOT included here -- it is used nowhere else, so there is
  nothing to de-duplicate, and merging it into either family above would risk
  silently changing its behavior. Leave it in place.

  Hartigan dip test: only the MC-synthetic family duplicated this test
  identically across three files, so only that one is shared here
  (apply_hartigan_dip_test). Fish_Modality_Tests/fish_hartigan_jitter_sweep.py
  calls `diptest.diptest(sample, sort_x=True)` directly, appears nowhere
  else, and is left as its own inline call.
"""

import numpy as np
from scipy.signal import find_peaks


# ══════════════════════════════════════════════════════════════════════════════
# MC-SYNTHETIC FAMILY
# Identical (byte-for-byte, modulo one docstring wording tweak) across
# MC_Cloud/mc_jpse_comprehensive.py, MC_Cloud_Bimodal/mc_bimodal_comprehensive.py,
# and MC_Cloud_Empirical/bootstrap_empirical.py before this consolidation.
# ══════════════════════════════════════════════════════════════════════════════

def _count_kde_modes(x_sample, bandwidth, grid_size=2000):
    """Count modes in a KDE with given bandwidth."""
    from scipy.stats import norm as _norm
    x_sorted = np.sort(x_sample)
    x_min, x_max = x_sorted[0], x_sorted[-1]
    x_grid = np.linspace(x_min - 2 * bandwidth, x_max + 2 * bandwidth, grid_size)
    z = (x_grid[:, None] - x_sorted[None, :]) / bandwidth
    pdf_vals = np.mean(_norm.pdf(z), axis=1)
    peaks, _ = find_peaks(pdf_vals, prominence=1e-4)
    return len(peaks)


def apply_silverman_test(x_sample, n_bootstrap=199, k=1, grid_size=2000):
    """
    Silverman (1981) critical-bandwidth test for unimodality (H0: <=k modes).
    Hall & York (2001) smooth-bootstrap calibration.
    Returns p-value (low -> reject H0 -> multimodal).
    """
    from scipy.stats import gaussian_kde
    try:
        x_sample = np.asarray(x_sample)
        n = len(x_sample)
        std_dev = np.std(x_sample, ddof=1)
        h0 = 1.06 * std_dev * n ** (-0.2)

        def find_critical_bandwidth(data):
            h_min, h_max = 0.01 * h0, 5.0 * h0
            if _count_kde_modes(data, h_min, grid_size) <= k:
                return h_min
            for _ in range(30):
                h_mid = (h_min + h_max) / 2
                if _count_kde_modes(data, h_mid, grid_size) > k:
                    h_min = h_mid
                else:
                    h_max = h_mid
            return (h_min + h_max) / 2

        h_crit_obs = find_critical_bandwidth(x_sample)
        if h_crit_obs < 0.1 * h0:
            return 0.99

        h_null = h0
        if _count_kde_modes(x_sample, h0, grid_size) > k:
            h_null = 1.5 * h_crit_obs
        kde_null = gaussian_kde(x_sample, bw_method=h_null / std_dev)

        h_crit_boot = np.zeros(n_bootstrap)
        for b in range(n_bootstrap):
            boot_sample = kde_null.resample(n).flatten()
            h_crit_boot[b] = find_critical_bandwidth(boot_sample)

        p_value = np.mean(h_crit_boot >= h_crit_obs)
        p_value = max(p_value, 1.0 / (n_bootstrap + 1))
        return float(p_value)
    except Exception:
        return None


def apply_hartigan_dip_test(x_sample):
    """Hartigan dip test. Returns p-value or None."""
    try:
        import diptest
        _, p_value = diptest.diptest(x_sample)
        return float(p_value)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FISH-JITTER FAMILY
# Identical (modulo type hints / n_inner default) between
# Fish_Modality_Tests/fish_silverman_jitter_sweep.py and
# fish_silverman_jitter_sweep_cloud.py before this consolidation.
# ══════════════════════════════════════════════════════════════════════════════

JITTER_N_GRID     = 128   # KDE evaluation grid points
JITTER_PROMINENCE = 0.05  # Peak prominence threshold as fraction of max PDF value


def _jitter_kde_pdf(data, h, n_grid=JITTER_N_GRID):
    """Evaluate Gaussian KDE on a uniform grid. Returns (n_grid,) PDF array."""
    x_min = data.min() - 3.0 * h
    x_max = data.max() + 3.0 * h
    x_grid = np.linspace(x_min, x_max, n_grid)
    diff = (x_grid[:, None] - data[None, :]) / h          # (n_grid, N)
    pdf  = np.exp(-0.5 * diff ** 2).sum(axis=1)            # unnormalised (fine for mode counting)
    return pdf


def jitter_count_modes(data, h, n_grid=JITTER_N_GRID):
    """Number of modes in Gaussian KDE of data at bandwidth h."""
    pdf = _jitter_kde_pdf(data, h, n_grid)
    prom = max(JITTER_PROMINENCE * pdf.max(), 1e-10)
    peaks, _ = find_peaks(pdf, prominence=prom)
    return max(1, len(peaks))


def jitter_find_h_crit(data, k=1, n_grid=JITTER_N_GRID, tol=1e-5):
    """
    Find h_crit = smallest bandwidth at which KDE(data, h) has <= k modes.
    Uses bisection between h_low (multimodal) and h_high (unimodal).

    Returns h_crit (the crossover bandwidth from multimodal to unimodal).
    If data is already unimodal at very small h, returns a small h_fallback.
    """
    n   = len(data)
    std = data.std(ddof=1)
    if std < 1e-10:
        return 1e-4   # constant data -- trivially unimodal

    # Silverman's rule of thumb as a starting upper bound
    h_max = 0.9 * std * n ** (-0.2)

    # Expand h_max until KDE is unimodal (at most <= k modes)
    for _ in range(30):
        if jitter_count_modes(data, h_max, n_grid) <= k:
            break
        h_max *= 2.0

    # h_min: small enough to be multimodal
    h_min = h_max * 0.001
    for _ in range(50):
        if jitter_count_modes(data, h_min, n_grid) > k:
            break
        h_min /= 2.0
        if h_min < 1e-8:
            # Data may be genuinely unimodal at all scales -- return small h
            return h_max * 0.001

    # Bisection
    for _ in range(60):
        h_mid = (h_min + h_max) / 2.0
        if jitter_count_modes(data, h_mid, n_grid) <= k:
            h_max = h_mid
        else:
            h_min = h_mid
        if (h_max - h_min) < tol * h_max:
            break

    return h_max   # smallest h with <= k modes


def jitter_silverman_pvalue(data, h_crit, k=1, n_inner=100, rng=None, n_grid=JITTER_N_GRID):
    """
    Bootstrap p-value for Silverman's test.

    Generates n_inner variance-preserving bootstrap samples from KDE(data, h_crit)
    and computes the fraction with h_crit* >= h_crit.

    Variance preservation (Hall & York 2001):
        x* = mu + (X_{b(i)} - mu + h_crit * Z_i) / sqrt(1 + h_crit^2 / s^2)
    where b(i) is a bootstrap index and Z_i ~ N(0,1).
    """
    if rng is None:
        rng = np.random.default_rng()

    n   = len(data)
    mu  = data.mean()
    s2  = data.var(ddof=1)
    c   = np.sqrt(1.0 + h_crit ** 2 / max(s2, 1e-10))

    exceed = 0
    for _ in range(n_inner):
        idx    = rng.integers(0, n, size=n)
        noise  = rng.standard_normal(size=n) * h_crit
        boot   = mu + (data[idx] - mu + noise) / c
        h_boot = jitter_find_h_crit(boot, k=k, n_grid=n_grid)
        if h_boot >= h_crit:
            exceed += 1

    return exceed / n_inner
