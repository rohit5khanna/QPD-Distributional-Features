"""
The paper's conventions and seeds, in one place.

*Bump Hunting with QPDs* is the interactive companion to Khanna & Bickel,
*Bump Hunting, Structural Overfitting, and Quantile-Parameterized
Distributions*. For that to mean anything, the notebook's DEFAULTS have to
reproduce the paper's numbers -- a reader should be able to open it, touch
nothing, and see the published result. Every control still moves; the paper is
simply where each one starts.

This module is the single source of truth for that. It mirrors the
`*_common.py` modules in the paper's `reproduction/` folder, so the notebook
and the reproduction scripts cannot drift apart again.

WHAT WAS WRONG BEFORE (and is fixed by using this module)

1. MONTE CARLO SAMPLES CAME FROM A DIFFERENT GENERATOR PATH. The notebook drew
   `dist.quantile(rng.random(n))` from one `default_rng(424_242)` consumed in a
   loop. The paper draws `jpse.JohnsonSU(...).rvs(size=n, random_state=seed)`
   with `seed = 42 + sample_size*1000 + replication`. jpse's `rvs` uses the
   legacy `np.random.seed` + inverse-CDF path, so the two produce DIFFERENT
   samples from the same nominal seed. No amount of matching N and K would
   have made the numbers agree.

2. THE BOOTSTRAP SECTION DEFAULTED TO THE WRONG REALIZATION. Seed 200612 was
   the default; the paper's pinned realization is 200043. 200612 is the
   realization the paper's own pipeline used in one place and not another --
   the inconsistency that was found and corrected during the reproduction work.

3. W1 WAS NOT EQUATION 6. The notebook integrated |Q_F - Q_T| over
   [0.001, 0.999] and did not normalize. Equation 6 is the MEAN over
   [0.01, 0.99], divided by the target's interdecile range Q_T(0.9) - Q_T(0.1).

4. THE GEYSER FILE WAS DEFECTIVE. `public/geyser.txt` held 298 rows: it was
   missing the first observation and recorded one waiting time as 55 where
   Azzalini & Bowman (1990) give 77. The canonical 299-point series is now in
   `public/geyser_299.csv`.

Anything in here that the manuscript states only implicitly is spelled out in
the constant's comment, so a reader can check it against the text.
"""
import numpy as np

# ---------------------------------------------------------------------------
# Conventions that apply everywhere in the paper
# ---------------------------------------------------------------------------

#: Equation 3 -- Weibull plotting positions p_i = i/(N+1).
def weibull(n):
    return np.arange(1, n + 1) / (n + 1)


#: The probability grid every W1 in the paper is evaluated on.
#: An empirical quantile function is undefined outside [p_1, p_N], which is why
#: the integral in Equation 6 runs over [0.01, 0.99] rather than literally [0, 1].
P_GRID = np.linspace(0.01, 0.99, 1000)

#: Bootstrap replicates per experiment, throughout the paper.
N_BOOT = 1000


def interdecile(qf_on_grid, grid=P_GRID):
    """Q_T(0.9) - Q_T(0.1): Equation 6's divisor, taken from the TARGET."""
    return float(np.interp(0.9, grid, qf_on_grid) - np.interp(0.1, grid, qf_on_grid))


def empirical_qf(x_sorted, grid=P_GRID):
    """EQF of a sample, at Weibull positions, on the W1 grid."""
    x_sorted = np.sort(np.asarray(x_sorted, float))
    return np.interp(grid, weibull(len(x_sorted)), x_sorted)


def w1(qf_on_grid, target_on_grid, divisor=None):
    """Returns (unnormalized, Equation 6).

    Equation 6:  W1 = mean |Q_F(p) - Q_T(p)|  /  (Q_T(0.9) - Q_T(0.1))
    over p in [0.01, 0.99].  Pass `divisor` (the TARGET's interdecile range) to
    get the normalized form; the fit's own spread is NOT the divisor.
    """
    raw = float(np.mean(np.abs(np.asarray(qf_on_grid, float)
                               - np.asarray(target_on_grid, float))))
    return raw, (raw / divisor if divisor else float('nan'))


# ---------------------------------------------------------------------------
# Section 1 -- Johnson Monte Carlo (Tables 3, A1; Figures 2, 3, A1-A4)
# ---------------------------------------------------------------------------

#: Shared by all three reference distributions (SU, SL, SB).
DIST_PARAMS = dict(eta=0, kappa=1, c=0.5, d=1.2)

MC_BASE_SEED = 42


def mc_seed(sample_size, replication, base=None):
    """The paper's Monte Carlo seed formula. Verified bit-exact against the
    stored checkpoints for 200/200 replications of Johnson SU at N = 200.

    `base` overrides MC_BASE_SEED so an interactive caller can draw a DIFFERENT
    but equally reproducible ensemble. base=42 is the paper.
    """
    return (MC_BASE_SEED if base is None else int(base)) + sample_size * 1000 + replication


def mc_draw(dist, sample_size, replication, base=None):
    """One Monte Carlo sample, by the paper's generator path.

    MUST go through jpse's `rvs`. `dist.quantile(rng.random(n))` gives a
    different sample for the same seed -- that is the single biggest reason the
    notebook's Monte Carlo numbers did not match the paper's.
    """
    return np.sort(np.asarray(
        dist.rvs(size=sample_size,
                 random_state=mc_seed(sample_size, replication, base)),
        float))


# ---------------------------------------------------------------------------
# Section 2 -- Bootstrap diagnostics (Table 4; Figures 4-7)
# ---------------------------------------------------------------------------

#: The paper's pinned realization: Johnson SU, N = 200, replication 1.
#: 200043 = 42 + 200*1000 + 1, i.e. mc_seed(200, 1).
BOOTSTRAP_SEED = 200043
BOOTSTRAP_N = 200
BOOTSTRAP_K = [4, 7, 10, 13]      # the paper's standard order set


def bootstrap_resample_seed(b):
    """Per-replicate seed, so replicate b is reachable without replaying a loop."""
    return BOOTSTRAP_SEED * 10_000 + b


def bootstrap_resample(x_unsorted, b):
    """Resample b of the pinned realization.

    Takes the sample in GENERATION order, not sorted order: `rng.choice`
    selects by index, so sorting first would silently change the draw.
    """
    rng = np.random.default_rng(bootstrap_resample_seed(b))
    return rng.choice(np.asarray(x_unsorted, float),
                      size=len(x_unsorted), replace=True)


# ---------------------------------------------------------------------------
# Sections 3-5 -- the empirical datasets
#
# Each gets its own stream base so no two sections can ever share draws. The
# ranges are disjoint by construction and asserted in `check_seed_families()`.
# ---------------------------------------------------------------------------

FISH_BASE = 0                 # replicate seeds 42 .. 1_001_041
FISH_STREAM_BASE = 5_000_000  # one-off jitter realizations
GEYSER_BASE = 20_000_000
GEYSER_STREAM_BASE = 25_000_000
HYDRO_BASE = 40_000_000
BIMODAL_BASE = 60_000_000

# Streams that belong to the NOTEBOOK, not the paper: the interactive
# "draw again" buttons and the asset-return section, which is the notebook's
# own extension. They were bare literals (70_000 for the bimodal redraw,
# 471_000 * 10_000 for the returns bootstrap). 70_000 was not merely
# undocumented, it was inside the fish pool: fish_seed(0.7, b) spans
# 70_042..71_041, so a redraw could land on a fish replicate's stream and
# check_seed_families would not have seen it, because it only checks the
# families it is told about. Both now have bases of their own, above every
# paper family, and are asserted alongside them.
RETURNS_BASE = 80_000_000
REDRAW_BASE = 90_000_000

#: Jitter is the HALF-WIDTH: jitter j means Uniform(-j, +j).
FISH_JITTER = 0.5             # lb   -- recorded weights are 1-lb heaped
GEYSER_JITTER = 0.5           # min  -- waiting times are whole minutes
HYDRO_JITTER = 0.0            # gauge heights are continuous; no jitter
#: Clip floor after jittering, per dataset -- matches FLOOR in the
#: reproduction's fish_common / geyser_common / hydrology_common.
FISH_FLOOR = 0.01
GEYSER_FLOOR = 1e-9
HYDRO_FLOOR = 1e-9


def _replicate(x_raw, jitter, floor, rng_seed):
    """One bootstrap replicate: resample with replacement, then jitter.

    `b` and `base` used to be parameters here and were never read -- the seed
    was always passed in ready-made. They are gone, so a caller cannot believe
    it is choosing a stream by passing them.
    """
    n = len(x_raw)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=n)
    s = np.asarray(x_raw, float)[idx]
    if jitter > 0:
        s = s + rng.uniform(-jitter, jitter, size=n)
    return np.clip(s, floor, None)


# ARGUMENT ORDER IS (jitter, b) IN ALL THREE. It used to be (jitter, b) for
# fish and (b, jitter) for geyser, with hydrology taking b alone. Both
# arguments are numbers, so geyser_seed(0.5, 3) against the old signature
# meant b=0.5, jitter=3 and returned a different stream with no error. The
# reproduction's *_common.py modules use (jitter, b); so does this module now.
# Jitter keeps its per-dataset default, so callers may still pass b alone --
# AS A KEYWORD.

def fish_seed(jitter=FISH_JITTER, b=0):
    return MC_BASE_SEED + FISH_BASE + int(round(jitter * 100)) * 10_000 + b


def fish_resample(x_raw, jitter=FISH_JITTER, b=0):
    return _replicate(x_raw, jitter, FISH_FLOOR, fish_seed(jitter, b))


def geyser_seed(jitter=GEYSER_JITTER, b=0):
    return MC_BASE_SEED + GEYSER_BASE + int(round(jitter * 100)) * 10_000 + b


def geyser_resample(x_raw, jitter=GEYSER_JITTER, b=0):
    return _replicate(x_raw, jitter, GEYSER_FLOOR, geyser_seed(jitter, b))


def hydro_seed(jitter=HYDRO_JITTER, b=0):
    return MC_BASE_SEED + HYDRO_BASE + int(round(jitter * 100)) * 10_000 + b


def hydro_resample(x_raw, jitter=HYDRO_JITTER, b=0):
    return _replicate(x_raw, jitter, HYDRO_FLOOR, hydro_seed(jitter, b))


def jitter_original(x_raw, jitter, base_stream, draw=0, floor=None):
    """One jitter realization of the RECORDED values -- no resampling.

    Used for point-estimate curves (every observation appears once). It needs
    its own stream: borrowing a replicate's seed would silently make the point
    estimate a duplicate of one bootstrap band member.

    CLIPS AT `floor`, as fish_common.jitter_original and
    geyser_common.jitter_original do. This module's version did not, which was
    a silent divergence rather than a numeric one -- at the paper's jitters the
    floor never binds (fish minimum is 1 lb against a 0.01 floor, geyser 43 min
    against 1e-9) -- but it would have bitten the moment anyone raised the
    jitter slider far enough to push a value to or below zero.
    """
    rng = np.random.default_rng(MC_BASE_SEED + base_stream
                                + int(round(jitter * 100)) * 10_000 + draw)
    x = np.asarray(x_raw, float)
    if jitter > 0:
        x = x + rng.uniform(-jitter, jitter, size=len(x))
    return x if floor is None else np.clip(x, floor, None)


# ---------------------------------------------------------------------------
# Section 6 -- bimodal recovery (Table 5, Figure 8)
# ---------------------------------------------------------------------------

BIMODAL_N = 200
BIMODAL_WEIGHTS = (0.60, 0.40)
BIMODAL_SEPARATION = -3.5     # x sigma of the base distribution, LEFTWARD.
                              # The sign is load-bearing: the base SU is
                              # skewed (c = 0.5), so a rightward shift is NOT
                              # the mirror image -- it is a different
                              # population, W1 = 0.589 away from this one.


def base_sigma(eta=0.0, kappa=1.0, c=0.5, d=1.2):
    """Closed-form standard deviation of the Johnson SU base distribution.

    Q(p) = eta + kappa*sinh((z - c)/d) with z = Phi^-1(p), so X = eta +
    kappa*sinh(W), W ~ Normal(mu, s^2) with mu = -c/d and s = 1/d. Using
    E[e^W] = e^{mu + s^2/2}:

        E[sinh W]   = e^{s^2/2} * sinh(mu)
        E[sinh^2 W] = (e^{2 s^2} * cosh(2 mu) - 1) / 2
        Var[X]      = kappa^2 * (E[sinh^2 W] - E[sinh W]^2)

    Exact -- no grid, no draws, no seed.  Two WRONG values were in use before:
    the notebook took np.std of the quantile function on a 5000-point grid
    over [0.0005, 0.9995] (1.3244, 3.26 % low, because that truncation loses
    real tail), and the paper's generator took np.std of 10,000 rvs draws
    under default_rng(9999) (1.3792, 0.74 % high). The manuscript says the
    second component sits 3.5 standard deviations away; only this value makes
    that sentence true.
    """
    mu, s2 = -c / d, 1.0 / d ** 2
    m1 = np.exp(s2 / 2) * np.sinh(mu)
    m2 = (np.exp(2 * s2) * np.cosh(2 * mu) - 1) / 2
    return float(kappa * np.sqrt(m2 - m1 ** 2))


BIMODAL_BASE_SIGMA = base_sigma(**DIST_PARAMS)   # 1.3690932625389083


def bimodal_shift(separation=None, sigma=None):
    """The offset applied to the second component, in x units."""
    sep = BIMODAL_SEPARATION if separation is None else separation
    return sep * (BIMODAL_BASE_SIGMA if sigma is None else sigma)


def bimodal_seed(replication):
    return MC_BASE_SEED + BIMODAL_BASE + replication


def bimodal_draw(dist, n, seed_base, offset, weight_a=None):
    """One sample from a two-component mixture, by the paper's sampler.

    Mirrors generate_bimodal_source.sample_bimodal_mixture (and therefore
    reproduction/scripts/bimodal_common.sample_mixture): the component split is
    a single binomial on the mixture rng, then each component is drawn by
    jpse's `rvs` on its own offset seed. `rng.random(n) < w` per draw plus
    inverse-CDF is an equally valid sampler and gives a DIFFERENT sample from
    the same seed, which is why it cannot be used here.
    """
    w = BIMODAL_WEIGHTS[0] if weight_a is None else float(weight_a)
    n1 = int(np.random.default_rng(seed_base).binomial(n, w))
    a = np.asarray(dist.rvs(size=n1, random_state=seed_base + 1000), float)
    b = np.asarray(dist.rvs(size=n - n1, random_state=seed_base + 2000), float) + offset
    return np.sort(np.concatenate([a, b]))


# ---------------------------------------------------------------------------
# Mode ranking and dispersion -- the conventions behind the paper's mode-IQR
# columns. THEY ARE NOT UNIFORM ACROSS SECTIONS, and the difference is
# deliberate, so they are separate functions rather than one with a flag:
#
#   hydrology (Table 6)  primary = the TALLEST peak            (hydrology_tables.py)
#   fish      (Table 9)  primary = the TALLEST peak            (fish_summary.py::_rank_modes)
#   geyser    (Table 11) primary = the peak at the LONGER WAIT (geyser_tables.py)
#
# Denominators are the same in all three: the primary columns are summarised
# over valid fits with at least ONE mode, the secondary columns over valid fits
# with at least TWO. (Verified against Table10_geyser_mode_iqr.csv, where
# n_primary at Log Metalog K=4 equals the 159 feasible fits and n_secondary is
# the 3 bimodal ones.)
# ---------------------------------------------------------------------------

def iqr(v):
    """Q3 - Q1, RAW (not normalized by the median). The paper reports raw IQRs."""
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)], float)
    if v.size == 0:
        return float('nan')
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 - q1)


def rank_modes_by_height(locs, hgts):
    """(primary_loc, primary_height, secondary_loc, secondary_height).

    PRIMARY is the tallest peak, SECONDARY the next tallest. Used by the fish
    and hydrology tables. A one-mode fit has no secondary and returns NaN there,
    which is why the secondary columns are summarised over a smaller set.
    """
    if locs is None or hgts is None or len(locs) == 0:
        return (float('nan'),) * 4
    locs = np.asarray(locs, float); hgts = np.asarray(hgts, float)
    order = np.argsort(hgts)[::-1]
    i = order[0]
    if len(order) == 1:
        return (float(locs[i]), float(hgts[i]), float('nan'), float('nan'))
    j = order[1]
    return (float(locs[i]), float(hgts[i]), float(locs[j]), float(hgts[j]))


def rank_modes_by_position(locs, hgts):
    """PRIMARY is the mode at the LARGEST x (the longer waiting time), SECONDARY
    the mode at the SMALLEST x (the shorter one) -- the two EXTREMES, not the
    top two.

    For a two-mode fit the two readings coincide, which is why this is easy to
    get wrong. They part company on fits with three or more modes: taking the
    "second longest" wait instead of the shortest gives a K=8 secondary-location
    IQR of 0.9365 against the published 0.6469, because 135 of the 719 bimodal-
    or-more fits at that order have a middle mode. Checked against
    Table10_geyser_mode_iqr.csv, which is what the manuscript's Table 11 prints.
    """
    if locs is None or hgts is None or len(locs) == 0:
        return (float('nan'),) * 4
    locs = np.asarray(locs, float); hgts = np.asarray(hgts, float)
    i = int(np.argmax(locs))
    if locs.size == 1:
        return (float(locs[i]), float(hgts[i]), float('nan'), float('nan'))
    j = int(np.argmin(locs))
    return (float(locs[i]), float(hgts[i]), float(locs[j]), float(hgts[j]))


def modality_split(n_modes):
    """(pct_1_mode, pct_2_modes, pct_gt2) over the fits handed in.

    PAPER CONVENTION: a 0-mode fit counts in the "1 mode %" column -- the
    manuscript's unimodal category is n_modes <= 1, not n_modes == 1. Fits with
    n_modes < 0 (unusable PDF) must be excluded by the CALLER; they are not part
    of the denominator.
    """
    v = np.asarray(n_modes, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return (float('nan'),) * 3
    return (100.0 * float((v <= 1).mean()),
            100.0 * float((v == 2).mean()),
            100.0 * float((v > 2).mean()))


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def check_seed_families(n_boot=N_BOOT, verbose=False):
    """Assert no two sections can draw the same stream. Cheap; run at import
    time in tests, not in the notebook."""
    fj = [round(0.05 * i, 2) for i in range(21)]
    pools = {
        'fish': {fish_seed(j, b) for j in fj for b in range(n_boot)}
                | {MC_BASE_SEED + FISH_STREAM_BASE + int(round(j * 100)) * 10_000 + d
                   for j in fj for d in range(100)},
        'geyser': {geyser_seed(b=b) for b in range(n_boot)}
                  | {MC_BASE_SEED + GEYSER_STREAM_BASE
                     + int(round(GEYSER_JITTER * 100)) * 10_000 + d for d in range(100)},
        'hydrology': {hydro_seed(b=b) for b in range(n_boot)},
        'bimodal': {bimodal_seed(r) for r in range(n_boot)}
                   | {bimodal_seed(r) + off for r in range(n_boot)
                      for off in (1000, 2000)},
        'returns': {RETURNS_BASE + b for b in range(n_boot)},
        'redraw': {REDRAW_BASE + d for d in range(n_boot)},
    }
    # The bimodal component seeds are seed_base + 1000 / + 2000, so replicate
    # r's SECOND component shares a stream with replicate (r + 1000)'s FIRST.
    # That is safe only while n_boot <= 1000, which is why it is asserted
    # rather than assumed -- raising N_BOOT would silently correlate them.
    assert len(pools['bimodal']) == 3 * n_boot, (
        'bimodal component streams overlap: replicate r component B shares a '
        'seed with replicate r+1000 component A. Raise the 1000/2000 offsets '
        f'before running more than 1000 replicates (n_boot={n_boot}).')

    names = list(pools)
    for i in range(len(names)):
        for j_ in range(i + 1, len(names)):
            clash = pools[names[i]] & pools[names[j_]]
            assert not clash, f'seed collision: {names[i]} vs {names[j_]}: {sorted(clash)[:5]}'
    if verbose:
        for k, v in pools.items():
            print(f'  {k:<10} [{min(v)}, {max(v)}]  n={len(v)}')
    return True


if __name__ == '__main__':
    print(f'P_GRID {P_GRID[0]:.3f}..{P_GRID[-1]:.3f} ({len(P_GRID)} pts), N_BOOT {N_BOOT}')
    print(f'MC seed for (N=200, rep=1): {mc_seed(200, 1)}  '
          f'-> bootstrap realization {BOOTSTRAP_SEED}  '
          f'{"OK" if mc_seed(200, 1) == BOOTSTRAP_SEED else "MISMATCH"}')
    print('seed families disjoint:', check_seed_families(verbose=True))
