"""
The paper's conventions and seeds, in one place.

QPD Playground is meant to be an interactive companion to Khanna & Bickel,
*Inferring Distributional Features based on Quantile-Parameterized
Distribution Fits*. For that to mean anything, the notebook's DEFAULTS have to
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


def mc_seed(sample_size, replication):
    """The paper's Monte Carlo seed formula. Verified bit-exact against the
    stored checkpoints for 200/200 replications of Johnson SU at N = 200."""
    return MC_BASE_SEED + sample_size * 1000 + replication


def mc_draw(dist, sample_size, replication):
    """One Monte Carlo sample, by the paper's generator path.

    MUST go through jpse's `rvs`. `dist.quantile(rng.random(n))` gives a
    different sample for the same seed -- that is the single biggest reason the
    notebook's Monte Carlo numbers did not match the paper's.
    """
    return np.sort(np.asarray(
        dist.rvs(size=sample_size, random_state=mc_seed(sample_size, replication)),
        float))


# ---------------------------------------------------------------------------
# Section 2 -- Bootstrap diagnostics (Table 4; Figures 4-7)
# ---------------------------------------------------------------------------

#: The paper's pinned realization: Johnson SU, N = 200, replication 1.
#: 200043 = 42 + 200*1000 + 1, i.e. mc_seed(200, 1).
BOOTSTRAP_SEED = 200043
BOOTSTRAP_N = 200
BOOTSTRAP_K = [4, 7, 10]


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

#: Jitter is the HALF-WIDTH: jitter j means Uniform(-j, +j).
FISH_JITTER = 0.5             # lb   -- recorded weights are 1-lb heaped
GEYSER_JITTER = 0.5           # min  -- waiting times are whole minutes
HYDRO_JITTER = 0.0            # gauge heights are continuous; no jitter
FISH_FLOOR = 0.01
GEYSER_FLOOR = 1e-9


def _replicate(x_raw, b, jitter, base, floor, rng_seed=None):
    n = len(x_raw)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, n, size=n)
    s = np.asarray(x_raw, float)[idx]
    if jitter > 0:
        s = s + rng.uniform(-jitter, jitter, size=n)
    return np.clip(s, floor, None)


def fish_seed(jitter, b):
    return MC_BASE_SEED + FISH_BASE + int(round(jitter * 100)) * 10_000 + b


def fish_resample(x_raw, jitter, b):
    return _replicate(x_raw, b, jitter, FISH_BASE, FISH_FLOOR, fish_seed(jitter, b))


def geyser_seed(b, jitter=GEYSER_JITTER):
    return MC_BASE_SEED + GEYSER_BASE + int(round(jitter * 100)) * 10_000 + b


def geyser_resample(x_raw, b, jitter=GEYSER_JITTER):
    return _replicate(x_raw, b, jitter, GEYSER_BASE, GEYSER_FLOOR, geyser_seed(b, jitter))


def hydro_seed(b):
    return MC_BASE_SEED + HYDRO_BASE + b


def hydro_resample(x_raw, b):
    return _replicate(x_raw, b, HYDRO_JITTER, HYDRO_BASE, 1e-9, hydro_seed(b))


def jitter_original(x_raw, jitter, base_stream, draw=0):
    """One jitter realization of the RECORDED values -- no resampling.

    Used for point-estimate curves (every observation appears once). It needs
    its own stream: borrowing a replicate's seed would silently make the point
    estimate a duplicate of one bootstrap band member.
    """
    rng = np.random.default_rng(MC_BASE_SEED + base_stream
                                + int(round(jitter * 100)) * 10_000 + draw)
    x = np.asarray(x_raw, float)
    if jitter > 0:
        x = x + rng.uniform(-jitter, jitter, size=len(x))
    return x


# ---------------------------------------------------------------------------
# Section 6 -- bimodal recovery (Table 5, Figure 8)
# ---------------------------------------------------------------------------

BIMODAL_N = 200
BIMODAL_WEIGHTS = (0.60, 0.40)
BIMODAL_SEPARATION = -3.5     # x sigma of the base distribution, leftward


def bimodal_seed(replication):
    return MC_BASE_SEED + BIMODAL_BASE + replication


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
        'geyser': {geyser_seed(b) for b in range(n_boot)}
                  | {MC_BASE_SEED + GEYSER_STREAM_BASE
                     + int(round(GEYSER_JITTER * 100)) * 10_000 + d for d in range(100)},
        'hydrology': {hydro_seed(b) for b in range(n_boot)},
        'bimodal': {bimodal_seed(r) for r in range(n_boot)},
    }
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
