# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "openpyxl",
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Bump Hunting with QPDs")


@app.cell
def _():
    import sys
    import os
    import io

    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _THIS_DIR)

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # NOTE: imported as `common.<pkg>`, not top-level `<pkg>`. marimo's WASM
    # export bundles the local `common/` folder as an installable wheel named
    # "common" (see common-*.whl), so once it's micropip-installed the only
    # importable path is `common.metalog`, `common.qflex`, etc. -- there is no
    # top-level `metalog`/`qflex`/`jpse`/`mode_utils` package in that build.
    # Importing this way also works locally, since `common/` sits right next
    # to app.py and _THIS_DIR is on sys.path (namespace package, no
    # common/__init__.py needed).
    from common.metalog.metalog_v2 import Metalog, MetalogError, LogMetalog, LogitMetalog
    from common.qflex.core import QFlex, ConstraintType
    from common.qflex.constraints import QFlexError
    from common.qflex.transforms import LogQFlex, LogitQFlex
    from common.jpse.johnson import JohnsonSU, JohnsonSL, JohnsonSB
    from common.mode_utils import detect_modes_from_arrays
    # Pure-Python/NumPy port of the Hartigan dip test, vendored (GPLv3,
    # see the header of common/dip_test.py) because the compiled `diptest`
    # PyPI package has no pure-Python wheel: a single unavailable wheel
    # aborts marimo's *entire* batched micropip install, breaking every
    # other dependency (plotly, scipy, ...) in the WASM/browser build, not
    # just this feature.
    from common.dip_test import dip_stat
    from common.dip_consts import Consts as DipConsts

    # The paper's conventions and seeds, in one module shared with the
    # paper's `reproduction/` scripts (common/paper_defaults.py). Every
    # default in this notebook comes from here, so "open it and touch
    # nothing" reproduces the published numbers; the controls still move.
    from common import paper_defaults as PAPER

    # Classical parametric reference models used as baselines alongside the
    # QPDs: a normal fit for the asset-return section and a GEV fit for the
    # river-gauge (annual block maxima) section, the latter matching the
    # paper's own GEV-vs-Log-Metalog comparison. Imported from the concrete
    # `scipy.stats` module so marimo's WASM static-import tracing sees it
    # (scipy is already a dependency of the QFlex solvers, so this adds no
    # new wheel to the browser build).
    from scipy.stats import norm as scipy_norm, genextreme as scipy_gev

    DATA_DIR = mo.notebook_location() / "public"
    return (
        ConstraintType,
        DATA_DIR,
        DipConsts,
        JohnsonSB,
        JohnsonSL,
        JohnsonSU,
        LogMetalog,
        LogQFlex,
        LogitMetalog,
        LogitQFlex,
        Metalog,
        MetalogError,
        QFlex,
        QFlexError,
        PAPER,
        detect_modes_from_arrays,
        dip_stat,
        go,
        io,
        make_subplots,
        mo,
        np,
        os,
        pd,
        scipy_gev,
        scipy_norm,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # Bump Hunting with QPDs

        A hands-on companion to *Bump Hunting, Structural Overfitting, and
        Quantile-Parameterized Distributions* (Khanna & Bickel). Every
        panel below is live: drag the sliders, click the buttons, and watch
        the Metalog and QFlex fits move &mdash; including the spurious modes
        that motivate the paper.

        **On this page:** Johnson distributions &middot; refitting under
        resampling (Monte Carlo and bootstrap, run as two separate
        experiments) &middot; a bimodal-mixture playground (likewise)
        &middot; four empirical case studies (each with its own QPD fit,
        Hartigan dip test, and bootstrap batch analysis). Every batch run
        reports all 4 QPDs &mdash; Metalog, QFlex-U, QFlex-TA+, QFlex-A+
        &mdash; with the paper's validity / false-modality / W1
        statistics, plus the Hartigan unimodality rejection rate.

        **How W₁ is measured, everywhere on this page.** The normalized
        Wasserstein-1 distance between a fitted quantile function $Q_F$ and its
        target $Q_T$, as the paper defines it in Equation 6:

        $$W_1 \;=\; \frac{1}{Q_T(0.9) - Q_T(0.1)}
        \int_0^1 \bigl|\,Q_F(p) - Q_T(p)\,\bigr| \; dp$$

        Evaluated here on $p \in [0.01,\, 0.99]$, since an empirical quantile
        function is undefined outside its smallest and largest plotting
        positions. Only the **target** changes from section to section: a Monte
        Carlo replicate is measured against its own sample, a bootstrap
        replicate against the reference sample it was drawn from, and an
        empirical section against the observed data. Each table names its
        target in the W₁ column heading.
        """
    )
    return


@app.cell
def _(
    ConstraintType,
    DipConsts,
    LogMetalog,
    LogQFlex,
    LogitMetalog,
    LogitQFlex,
    Metalog,
    MetalogError,
    QFlex,
    QFlexError,
    detect_modes_from_arrays,
    dip_stat,
    go,
    make_subplots,
    np,
    pd,
    scipy_gev,
    scipy_norm,
):
    # Shared helpers used by every fitting/plotting/batch-simulation section
    # below, so the core logic exists in exactly one place instead of being
    # copy-pasted per section.
    #
    # Deliberately stops short of p=0/1 exactly (numerical blowup there for
    # unbounded/unconstrained fits) but reaches much closer to the edges
    # than an earlier, more conservative (0.01, 0.99) grid did -- that
    # narrower grid left every fitted curve visibly short of the data's own
    # min/max, making PDFs look like they don't extend to the ends of the
    # plots. QFlex-U in particular can still spike or go infeasible out
    # here; render_empirical_panel's PDF-axis guard (below) keeps a single
    # spike from blowing out the whole y-axis when that happens.
    FIT_P_GRID = np.linspace(0.001, 0.999, 1000)

    # W1 IS NOT COMPUTED ON FIT_P_GRID. That grid is deliberately wide so the
    # drawn PDFs reach the edges of the plots. Equation 6 integrates over
    # [0.01, 0.99] -- an empirical quantile function is undefined outside
    # [p_1, p_N] -- and divides by the TARGET's interdecile range. Mixing the
    # two up is what made this notebook's W1 disagree with the paper's.
    W1_P_GRID = PAPER.P_GRID

    def _modes_paper_grid(fit):
        """Mode count on the PAPER's interval, [0.01, 0.99].

        The live panels draw on FIT_P_GRID, which reaches much closer to p=0/1
        so the curves span the plots. Counting modes out there disagrees with
        the paper -- mode_utils.detect_modes_in_pdf uses [0.01, 0.99], and the
        extreme tails are exactly where spurious bumps appear. Without this the
        live panel and the batch summary could report different mode counts for
        the same fit.
        """
        _x = np.asarray(fit.quantile(W1_P_GRID), float)
        _p = np.clip(np.asarray(fit.pdf(W1_P_GRID), float), 0, None)   # as the paper does
        return detect_modes_from_arrays(_x, _p)

    def paper_settings_note(mo_ref, seed_text, settings, note=None):
        """A compact "to reproduce the paper, use these" callout.

        Every section states the seed it draws with and the settings that
        reproduce the published numbers, so a reader who has moved a slider
        can get back to the paper without hunting through the source."""
        _rows = "\n".join(f"| {_k} | `{_v}` |" for _k, _v in settings)
        _extra = f"\n\n{note}" if note else ""
        return mo_ref.md(
            f"""
/// details | Reproducing the paper's numbers in this section
    type: info

**Seed:** {seed_text}

| setting | paper value |
|---|---|
{_rows}

These are the defaults, so an untouched notebook already reproduces the
paper — every value here is shared with the scripts that produced the
published tables. The **QFlex constraint** control drives only the single-fit
panel; the batch summary always fits all four QPDs, and **K** selects which
block of the paper's table a run reproduces rather than changing the
experiment.{_extra}
///
"""
        )

    # Paper convention: the constrained variants of QFlex get their own name,
    # not a generic "QFlex" label that hides which constraint was actually
    # used.
    QFLEX_LABELS = {"NONE": "QFlex-U", "A": "QFlex-A+", "TA": "QFlex-TA+"}

    # The 4 QPDs, in the order used consistently across this notebook and the
    # companion Returns-analysis workbook: Metalog, then the 3 QFlex
    # constraint variants with TA+ before A+.
    MODEL_ORDER = ("Metalog", "QFlex-U", "QFlex-TA+", "QFlex-A+")

    # Same per-model palette as the companion Returns-analysis workbook, so a
    # color means the same model whether you're looking at this notebook or
    # that spreadsheet. Previously every QFlex curve was plotted in the same
    # green no matter which constraint was active, which quietly discarded
    # the one piece of information (which constraint?) most worth encoding
    # visually -- the constraint dropdown's own text was the only way to
    # tell U/TA+/A+ apart at a glance.
    # These four are RESERVED for the four QPDs and are used for nothing else,
    # here or in the paper -- they are the same hex values as the paper's
    # reproduction/scripts/figure_style.py. Anything that is not one of these
    # models (GEV, the population curve, observed data, bootstrap bands) draws
    # a colour from outside this dict.
    MODEL_COLORS = {
        "Metalog": "#1f77b4",
        "QFlex-U": "#9467bd",
        "QFlex-TA+": "#ff7f0e",
        "QFlex-A+": "#2ca02c",
    }
    _QFLEX_CONSTRAINTS = {"QFlex-U": "NONE", "QFlex-TA+": "TA", "QFlex-A+": "A"}

    # Boundedness support -- matching the paper's own convention of fitting
    # semi-bounded (Log Metalog / Log QFlex, exponential transform) or
    # bounded (Logit Metalog / Logit QFlex, logit transform) QPDs wherever
    # the underlying quantity has a natural domain restriction, rather than
    # always using the unbounded variant. `bounds` is either None
    # (unbounded), (lower, None) (semi-bounded), or (lower, upper)
    # (bounded); every fitting call site below threads a `bounds` argument
    # through to these three helpers so the boundedness choice is made in
    # exactly one place per section.
    def _make_metalog(x, y, k, bounds):
        if bounds is None:
            return Metalog(x, y, terms=k)
        _lo, _hi = bounds
        if _hi is None:
            return LogMetalog(x, y, lower_bound=_lo, terms=k)
        return LogitMetalog(x, y, lower_bound=_lo, upper_bound=_hi, terms=k)

    def _make_qflex(x, y, k, constraint, bounds):
        if bounds is None:
            return QFlex(x, y, terms=k, constraint_type=constraint)
        _lo, _hi = bounds
        if _hi is None:
            return LogQFlex(x, y, lower_bound=_lo, terms=k, constraint_type=constraint)
        return LogitQFlex(x, y, lower_bound=_lo, upper_bound=_hi, terms=k, constraint_type=constraint)

    def _qpd_prefix(bounds):
        """'' for unbounded, 'Log ' for semi-bounded, 'Logit ' for bounded
        -- prepended to every displayed Metalog/QFlex model name so the plot
        legends, mode summaries, and batch tables always say which variant
        was actually fit, matching the paper's own Log Metalog / Log QFlex
        naming."""
        if bounds is None:
            return ""
        return "Log " if bounds[1] is None else "Logit "

    # Classical parametric reference fits, used as baselines against the
    # QPDs. The paper does exactly this for the hydrology case, fitting a
    # GEV to the annual block maxima and contrasting its (stable) implied
    # mode against the (unstable) high-order Log Metalog one; a normal fit
    # plays the same role for the asset returns. Both are returned in the
    # same {'name', 'color', 'curve', 'n_modes'} shape that
    # render_empirical_panel's `reference_fits` expects, with the curve
    # evaluated on FIT_P_GRID so W1 is computed on the same grid as every
    # QPD curve.
    # figure_style.OTHER in the reproduction: GEV is BROWN. It was sea green
    # (#2E8B57) here, one step from QFlex-A+ (#2ca02c) -- and the hydrology
    # panel draws GEV and Log QFlex-A+ on the same axes, so the two reference
    # curves the reader is meant to compare were near-indistinguishable.
    #: figure_style.OTHER['observed'] -- observed data drawn as MARKERS, as
    #: distinct from the empirical CURVE.
    OBSERVED_COLOR = "#40556b"

    REFERENCE_COLORS = {"Normal": "#B5651D", "GEV": "#8c564b"}

    def make_normal_reference(x_raw):
        """Maximum-likelihood normal fit (= sample mean / sd)."""
        _mu = float(np.mean(x_raw))
        _sd = float(np.std(x_raw, ddof=1))
        _xg = scipy_norm.ppf(FIT_P_GRID, loc=_mu, scale=_sd)
        _pg = scipy_norm.pdf(_xg, loc=_mu, scale=_sd)
        return {
            "name": "Normal", "color": REFERENCE_COLORS["Normal"],
            "curve": (_xg, _pg), "n_modes": 1,
            "params": {"mu": _mu, "sigma": _sd},
        }

    def make_gev_reference(x_raw):
        """Maximum-likelihood GEV fit. scipy's `genextreme` shape `c` is the
        negative of the usual EVT xi, so xi = -c is reported instead: xi > 0
        is Frechet (heavy tail), xi = 0 Gumbel, xi < 0 Weibull (bounded
        upper tail). The GEV density is unimodal by construction, which is
        the whole point of using it as the stable baseline here."""
        _c, _loc, _scale = scipy_gev.fit(np.asarray(x_raw, dtype=float))
        _xg = scipy_gev.ppf(FIT_P_GRID, _c, loc=_loc, scale=_scale)
        _pg = scipy_gev.pdf(_xg, _c, loc=_loc, scale=_scale)
        return {
            "name": "GEV", "color": REFERENCE_COLORS["GEV"],
            "curve": (_xg, _pg), "n_modes": 1,
            "params": {"xi": -float(_c), "loc": float(_loc), "scale": float(_scale)},
        }

    # Shared look for every figure in the notebook -- clean white
    # background, thin dark-gray box axes, no heavy gridlines, restrained
    # tick density -- matching the companion QFlex tutorial notebook's
    # matplotlib-default aesthetic instead of Plotly's stock "plotly"
    # template (a light blue-gray plot area with thick white gridlines,
    # which is what made every panel here look inconsistent/"off").
    # Applied via style_fig() right before each figure is returned, so
    # every plot in the notebook shares one visual identity.
    PLOT_FONT = dict(family="Helvetica, Arial, sans-serif", size=12, color="#262626")

    def style_fig(fig, dense_ticks=False):
        """Apply the notebook-wide plot style. Call this last, right
        before returning/displaying a figure -- it only touches
        background/font/axis-line/tick styling, never title_text, range,
        or other content-specific properties set earlier, so call order
        relative to those doesn't matter.

        dense_ticks=True tightens tick spacing (more ticks) for the
        Johnson-distribution panels (the standalone Johnson distributions
        plot, and every Monte Carlo / bimodal-mixture panel built from
        them via render_mc_panel) -- their smooth, low-curvature reference
        shapes read better with a finer grid than the empirical-data
        panels, which have plenty of visual detail from the histogram/raw
        points already."""
        fig.update_layout(
            template="simple_white",
            font=PLOT_FONT,
            title_font=dict(size=15, color="#1A1A1A"),
            legend=dict(font=dict(size=11.5)),
            hoverlabel=dict(font_size=12, font_family=PLOT_FONT["family"]),
        )
        _nticks_x = 11 if dense_ticks else 7
        _nticks_y = 9 if dense_ticks else 6
        fig.update_xaxes(
            showline=True, linewidth=1.3, linecolor="#4A4A4A", mirror=True,
            showgrid=True, gridcolor="#E7E9EE", gridwidth=1, zeroline=False,
            ticks="outside", ticklen=5, tickwidth=1.1, tickcolor="#4A4A4A",
            tickfont=dict(size=11.5), title_font=dict(size=13),
            nticks=_nticks_x,
        )
        fig.update_yaxes(
            showline=True, linewidth=1.3, linecolor="#4A4A4A", mirror=True,
            showgrid=True, gridcolor="#E7E9EE", gridwidth=1, zeroline=False,
            ticks="outside", ticklen=5, tickwidth=1.1, tickcolor="#4A4A4A",
            tickfont=dict(size=11.5), title_font=dict(size=13),
            nticks=_nticks_y,
        )
        fig.update_annotations(font=dict(size=13.5, color="#262626"))  # subplot titles
        return fig

    def section_header_html(text, level=2):
        """Raw HTML (for embedding into an f-string markdown block) for a
        visually prominent section header. Plain markdown `##`/`###` was
        easy to miss while scrolling -- every header rendered the same
        muted weight as the body text around it, so a new section didn't
        register. This adds real size contrast, a colored left accent bar,
        a soft background band, and generous top spacing, so a new section
        is unmistakable at a glance rather than something you have to
        notice you've entered. Level 2 = main section (blue accent, larger);
        level 3 = subsection (purple accent, smaller)."""
        _cfg = {
            2: dict(size="26px", weight=800, color="#16213A", mt="12px",
                     accent="#3B5FA0", bg="#EEF1F8"),
            3: dict(size="19px", weight=700, color="#2A2A2A", mt="6px",
                     accent="#8E5AA6", bg="#F5F1F7"),
        }[level]
        return (
            f'<div style="margin-top:{_cfg["mt"]}; margin-bottom:16px; '
            f'border-left:6px solid {_cfg["accent"]}; background:{_cfg["bg"]}; '
            f'padding:10px 16px; border-radius:0 4px 4px 0;">'
            f'<span style="font-size:{_cfg["size"]}; font-weight:{_cfg["weight"]}; '
            f'color:{_cfg["color"]}; letter-spacing:-0.01em;">{text}</span>'
            f'</div>'
        )

    def fit_all_qpds(x_sorted, y_plot_pos, k_metalog_val, k_qflex_val, bounds=None):
        """Fit all 4 QPDs (Metalog + all 3 QFlex constraint variants) to one
        (x, y) EQF sample. Never raises. Returns a dict keyed by MODEL_ORDER
        label -> {"fit", "curve", "modes"} (same shape per model), plus a
        combined error string (or None). Used wherever a batch/summary needs
        all 4 models at once, rather than just the single QFlex constraint
        shown in a section's live 2-model panel. `bounds` selects the
        boundedness variant -- see _qpd_prefix above."""
        _results = {}
        _errors = []
        try:
            _mf = _make_metalog(x_sorted, y_plot_pos, k_metalog_val, bounds)
            _xg, _pg = _mf.quantile(FIT_P_GRID), _mf.pdf(FIT_P_GRID)
            _results["Metalog"] = {"fit": _mf, "curve": (_xg, _pg), "modes": _modes_paper_grid(_mf)}
        except MetalogError as e:
            _results["Metalog"] = {"fit": None, "curve": None, "modes": (None, None, None)}
            _errors.append(f"Metalog: {e}")

        for _label in ("QFlex-U", "QFlex-TA+", "QFlex-A+"):
            try:
                _constraint = ConstraintType[_QFLEX_CONSTRAINTS[_label]]
                _qf = _make_qflex(x_sorted, y_plot_pos, k_qflex_val, _constraint, bounds)
                _xg, _pg = _qf.quantile(FIT_P_GRID), _qf.pdf(FIT_P_GRID)
                _results[_label] = {"fit": _qf, "curve": (_xg, _pg), "modes": _modes_paper_grid(_qf)}
            except QFlexError as e:
                _results[_label] = {"fit": None, "curve": None, "modes": (None, None, None)}
                _errors.append(f"{_label}: {e}")

        return _results, (" · ".join(_errors) if _errors else None)

    def hartigan_test(x):
        """Hartigan dip test for unimodality. Uses a vendored pure-Python/
        NumPy port of the algorithm (common/dip_test.py + common/
        dip_consts.py, GPLv3 -- see those files' headers) rather than the
        compiled `diptest` PyPI package, because that package has no
        pure-Python wheel and its absence would break marimo's WASM/browser
        build entirely (a single unavailable wheel aborts the whole batched
        dependency install, not just this feature). Cheap (no QPD fitting
        involved) -- unlike Silverman's test, this is fast enough to run on
        every replicate of a bootstrap batch, not just the single raw/
        current sample."""
        _x_sorted = np.sort(np.asarray(x, dtype=float))
        _n = len(_x_sorted)
        _dip = dip_stat(_x_sorted)
        # Matches diptest.diptest()'s own convention: the dip test isn't
        # valid for n <= 3, so the p-value is fixed at 1.0 (never rejects)
        # rather than extrapolating the critical-value table below its
        # tabulated range.
        _pval = 1.0 if _n <= 3 else float(DipConsts.compute_pval_interpolation(_n, _dip))
        return float(_dip), _pval, bool(_pval < 0.05)

    def hartigan_line_md(mo, x):
        """The one-line Hartigan verdict for a single (raw/current) sample:
        just reject-or-not, per the paper's convention -- the interesting
        aggregate is the *rate* of rejection across a bootstrap batch,
        reported separately by run_replicate_batch below."""
        _dip, _pval, _reject = hartigan_test(x)
        _verdict = "**rejects** unimodality" if _reject else "**rejects** spurious modes"
        return mo.md(
            f"**Hartigan dip test** (this sample, N={len(x)}): dip = {_dip:.4f}, p = {_pval:.4f} "
            f"&rarr; {_verdict} at α=0.05."
        )

    def fit_metalog_qflex(x_sorted, y_plot_pos, k_metalog_val, k_qflex_val, constraint_label, bounds=None):
        """Fit Metalog and QFlex to one (x, y) EQF sample. Never raises.
        `bounds` selects the boundedness variant -- see _qpd_prefix above."""
        _fit_error = None
        _metalog_fit = _qflex_fit = None
        _metalog_curve = _qflex_curve = None
        _metalog_modes = _qflex_modes = (None, None, None)

        try:
            _mf = _make_metalog(x_sorted, y_plot_pos, k_metalog_val, bounds)
            _xg = _mf.quantile(FIT_P_GRID)
            _pg = _mf.pdf(FIT_P_GRID)
            _metalog_fit = _mf
            _metalog_curve = (_xg, _pg)
            _metalog_modes = _modes_paper_grid(_mf)
        except MetalogError as e:
            _fit_error = f"Metalog: {e}"

        try:
            _constraint = ConstraintType[constraint_label]
            _qf = _make_qflex(x_sorted, y_plot_pos, k_qflex_val, _constraint, bounds)
            _xg = _qf.quantile(FIT_P_GRID)
            _pg = _qf.pdf(FIT_P_GRID)
            _qflex_fit = _qf
            _qflex_curve = (_xg, _pg)
            _qflex_modes = _modes_paper_grid(_qf)
        except QFlexError as e:
            _fit_error = (_fit_error + " · " if _fit_error else "") + f"{QFLEX_LABELS[constraint_label]}: {e}"

        return {
            "fit_error": _fit_error,
            "metalog_fit": _metalog_fit, "metalog_curve": _metalog_curve, "metalog_modes": _metalog_modes,
            "qflex_fit": _qflex_fit, "qflex_curve": _qflex_curve, "qflex_modes": _qflex_modes,
        }

    def mode_summary_md(mo, fit_error, metalog_fit, metalog_modes, qflex_fit, qflex_modes, constraint_label,
                          metalog_w1=None, qflex_w1=None, w1_label="empirical", bounds=None,
                          true_n_modes=1):
        """`true_n_modes` is how many modes the underlying population is
        actually believed to have, which is what makes a reported extra mode
        "spurious" or not. Calling every multi-mode fit spurious is only
        right where the truth is unimodal (the Johnson MC sections, the
        river gauge): in the bimodal-mixture playground two modes are the
        correct answer, and for the Old Faithful geyser two modes are well
        established, so labelling those "spurious structure" misreports the
        result. Pass None where the true modality is genuinely unresolved
        (the fish weights) and the count is reported with no verdict
        attached."""
        _prefix = _qpd_prefix(bounds)
        _metalog_name = f"{_prefix}Metalog"
        _qflex_name = _prefix + QFLEX_LABELS[constraint_label]

        def _shape_text(n_modes):
            if not n_modes:
                return "no modes detected"
            if n_modes == 1:
                return "unimodal" if (true_n_modes is None or true_n_modes == 1) else "1 mode — misses the true structure"
            if true_n_modes is None:
                return f"{n_modes} modes"
            if n_modes > true_n_modes:
                return f"{n_modes} modes — spurious structure"
            return f"{n_modes} modes — matches the true structure"

        def _line(name, fit, modes, w1):
            if fit is None:
                return f"**{name}:** fit failed"
            _feas = "valid" if fit.is_feasible else "⚠️ not valid (PDF goes negative)"
            _w1_txt = f" | **W1 vs {w1_label}** = {w1:.4f}" if w1 is not None else ""
            return f"**{name}:** {_feas}, {_shape_text(modes[0])}{_w1_txt}"

        return mo.md(
            f"{_line(_metalog_name, metalog_fit, metalog_modes, metalog_w1)}  \n"
            f"{_line(_qflex_name, qflex_fit, qflex_modes, qflex_w1)}"
            + (f"\n\n⚠️ {fit_error}" if fit_error else "")
        )

    def true_dist_ranges(dist, p_lo=0.001, p_hi=0.999, y_headroom=1.25, n_grid=800):
        """Fixed axis ranges derived only from the reference distribution's
        own shape -- NOT from any particular sample or K -- so re-drawing a
        sample or changing K doesn't rescale the axes out from under you."""
        _p = np.linspace(p_lo, p_hi, n_grid)
        _x = dist.quantile(_p)
        _pdf = dist.pdf(_x)
        _lo, _hi = float(np.min(_x)), float(np.max(_x))
        _span = max(_hi - _lo, 1e-9)
        x_range = [_lo - 0.04 * _span, _hi + 0.04 * _span]
        y_range = [0.0, float(np.max(_pdf)) * y_headroom]
        return x_range, y_range

    def render_mc_panel(mo, plotly_config, true_dist, x_sample, y_sample, k_metalog_val, k_qflex_val,
                          constraint_label, metalog_curve, metalog_fit, metalog_modes, qflex_curve, qflex_fit,
                          qflex_modes, fit_error, x_range, y_range, bounds=None, true_n_modes=1):
        """The QF + PDF panel pair shared by the Johnson MC section and the
        bimodal-mixture MC section: true curve, sample, both fits, both
        fits' true-curve overlay for contrast, and fixed axes. `bounds`
        selects the Metalog/QFlex variant (unbounded / semi-bounded / bounded)
        and is only used for display-name prefixing here (the actual fitting
        is done upstream in fit_metalog_qflex)."""
        _qflex_name = QFLEX_LABELS[constraint_label]
        _qflex_color = MODEL_COLORS[_qflex_name]
        _prefix = _qpd_prefix(bounds)
        _metalog_display = f"{_prefix}Metalog"
        _qflex_display = _prefix + _qflex_name
        _fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Quantile function", "Probability density"),
            horizontal_spacing=0.09,
        )

        _p_true = np.linspace(0.005, 0.995, 400)
        _x_true = true_dist.quantile(_p_true)
        _x_dense = np.linspace(x_range[0], x_range[1], 400)
        _pdf_true = true_dist.pdf(_x_dense)

        _fig.add_trace(
            go.Scatter(x=_p_true, y=_x_true, mode="lines", name="True QF",
                        line=dict(color="#9AA3B8", width=1.8, dash="dot")),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=_x_dense, y=_pdf_true, mode="lines", name="True PDF",
                        line=dict(color="#9AA3B8", width=1.8, dash="dot")),
            row=1, col=2,
        )
        _fig.add_trace(
            go.Scatter(x=y_sample, y=x_sample, mode="markers", name="Sample",
                        marker=dict(color=OBSERVED_COLOR, size=5, opacity=0.55)),
            row=1, col=1,
        )

        if metalog_curve is not None:
            _xg, _pg = metalog_curve
            _fig.add_trace(
                go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name=f"{_metalog_display} K={k_metalog_val}",
                            line=dict(color=MODEL_COLORS["Metalog"], width=2.4)),
                row=1, col=1,
            )
            _fig.add_trace(
                go.Scatter(x=_xg, y=_pg, mode="lines", name=f"{_metalog_display} K={k_metalog_val}",
                            line=dict(color=MODEL_COLORS["Metalog"], width=2.4), showlegend=False),
                row=1, col=2,
            )
            _n_modes, _locs, _hgts = metalog_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(
                    go.Scatter(x=_locs, y=_hgts, mode="markers", name=f"{_metalog_display} modes",
                                marker=dict(color=MODEL_COLORS["Metalog"], size=10, symbol="diamond",
                                            line=dict(color="white", width=1))),
                    row=1, col=2,
                )

        if qflex_curve is not None:
            _xg, _pg = qflex_curve
            _fig.add_trace(
                go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name=f"{_qflex_display} K={k_qflex_val}",
                            line=dict(color=_qflex_color, width=2.4)),
                row=1, col=1,
            )
            _fig.add_trace(
                go.Scatter(x=_xg, y=_pg, mode="lines", name=f"{_qflex_display} K={k_qflex_val}",
                            line=dict(color=_qflex_color, width=2.4), showlegend=False),
                row=1, col=2,
            )
            _n_modes, _locs, _hgts = qflex_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(
                    go.Scatter(x=_locs, y=_hgts, mode="markers", name=f"{_qflex_display} modes",
                                marker=dict(color=_qflex_color, size=10, symbol="diamond",
                                            line=dict(color="white", width=1))),
                    row=1, col=2,
                )

        _fig.update_xaxes(title_text="Cumulative probability", range=[0, 1], row=1, col=1)
        _fig.update_yaxes(title_text="Value", range=x_range, row=1, col=1)
        _fig.update_xaxes(title_text="Value", range=x_range, row=1, col=2)
        _fig.update_yaxes(title_text="Density", range=y_range, row=1, col=2)
        _fig.update_layout(
            height=460, margin=dict(l=55, r=25, t=45, b=75),
            legend=dict(orientation="h", y=-0.22), uirevision="keep-zoom",
        )
        style_fig(_fig, dense_ticks=True)

        # W1 is measured against THIS REPLICATE'S OWN SAMPLE, not against the
        # true distribution -- the paper's Monte Carlo tables (3 and A1) do the
        # same. Distance-to-truth is a different quantity (estimation error);
        # this one is fit to the data actually observed, which keeps the number
        # comparable with the empirical sections, where no truth exists.
        _eqf_on_w1 = PAPER.empirical_qf(x_sample, W1_P_GRID)
        _L = PAPER.interdecile(_eqf_on_w1)
        def _eq6(curve):
            if curve is None:
                return None
            _q = np.interp(W1_P_GRID, FIT_P_GRID, curve[0])
            return PAPER.w1(_q, _eqf_on_w1, _L)[1]
        _metalog_w1 = _eq6(metalog_curve)
        _qflex_w1 = _eq6(qflex_curve)

        return mo.vstack([
            mode_summary_md(mo, fit_error, metalog_fit, metalog_modes, qflex_fit, qflex_modes, constraint_label,
                              metalog_w1=_metalog_w1, qflex_w1=_qflex_w1, w1_label="sample", bounds=bounds,
                              true_n_modes=true_n_modes),
            mo.ui.plotly(_fig, config=plotly_config),
        ])

    def render_empirical_panel(mo, plotly_config, title, axis_label, constraint_label, p_grid, eqf_point, eqf_lo,
                                 eqf_hi, x_raw, metalog_curve, metalog_fit, metalog_modes, qflex_curve, qflex_fit,
                                 qflex_modes, fit_error, value_xlim=None, bounds=None, reference_fits=None,
                                 true_n_modes=1, bin_width=None, bin_start=None):
        """The EQF+CI / Metalog / QFlex panel pair shared by the four
        empirical dataset sections. The PDF panel also shows a histogram of
        the raw data for reference. `bounds` selects the boundedness
        variant (see _qpd_prefix above) and is reflected in every legend
        label and the mode summary below the plot.

        `reference_fits` optionally adds classical parametric baselines
        drawn alongside the QPDs -- a normal fit for the asset returns, a
        GEV fit for the river-gauge block maxima (the paper's own GEV
        comparison). Each entry is a dict with 'name', 'color' and 'curve'
        = (x_on_FIT_P_GRID, pdf_at_those_x); they are plotted as dashed
        lines to keep them visually distinct from the fitted QPDs, and
        their W1-vs-empirical distances are reported under the QPD lines
        so the comparison the paper makes is readable off the page."""
        _qflex_name = QFLEX_LABELS[constraint_label]
        _qflex_color = MODEL_COLORS[_qflex_name]
        _prefix = _qpd_prefix(bounds)
        _metalog_display = f"{_prefix}Metalog"
        _qflex_display = _prefix + _qflex_name
        _fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Quantile function", "Probability density"),
            horizontal_spacing=0.09,
        )

        _fig.add_trace(go.Scatter(
            x=np.concatenate([p_grid, p_grid[::-1]]),
            y=np.concatenate([eqf_hi, eqf_lo[::-1]]),
            fill="toself", fillcolor="rgba(59, 95, 160, 0.15)",
            line=dict(width=0), hoverinfo="skip", name="95% bootstrap CI",
        ), row=1, col=1)
        _fig.add_trace(go.Scatter(
            x=p_grid, y=eqf_point, mode="lines", name="Empirical QF",
            line=dict(color="#9AA3B8", width=1.8, dash="dot"),
        ), row=1, col=1)
        _n_raw = len(x_raw)
        _p_raw = np.arange(1, _n_raw + 1) / (_n_raw + 1)
        _stride = max(1, _n_raw // 400)
        _fig.add_trace(go.Scatter(
            x=_p_raw[::_stride], y=x_raw[::_stride], mode="markers", name="Raw data",
            marker=dict(color=OBSERVED_COLOR, size=4, opacity=0.35),
        ), row=1, col=1)

        # `bin_width` follows the paper: bins of 2 x the jitter half-width, so
        # one bin holds exactly the interval a recorded value can be smeared
        # over (1 lb for fish at +/-0.5 lb, 1 min for geyser at +/-0.5 min).
        # Sections that set no bin_width keep Plotly's 40-bin default.
        if bin_width:
            _hist_bins = dict(
                autobinx=False,
                xbins=dict(size=bin_width,
                           start=(float(np.min(x_raw)) - bin_width / 2.0
                                  if bin_start is None else bin_start)),
            )
        else:
            _hist_bins = dict(nbinsx=40)
        _fig.add_trace(go.Histogram(
            x=x_raw, histnorm="probability density", name="Data histogram",
            marker=dict(color="#C7CCDA"), opacity=0.55, **_hist_bins,
        ), row=1, col=2)

        if metalog_curve is not None:
            _xg, _pg = metalog_curve
            _fig.add_trace(go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name=f"{_metalog_display} fit",
                                        line=dict(color=MODEL_COLORS["Metalog"], width=2.4)), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_xg, y=_pg, mode="lines", name=f"{_metalog_display} fit",
                                        line=dict(color=MODEL_COLORS["Metalog"], width=2.4), showlegend=False), row=1, col=2)
            _n_modes, _locs, _hgts = metalog_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(go.Scatter(x=_locs, y=_hgts, mode="markers", name=f"{_metalog_display} modes",
                                            marker=dict(color=MODEL_COLORS["Metalog"], size=10, symbol="diamond",
                                                        line=dict(color="white", width=1))), row=1, col=2)

        if qflex_curve is not None:
            _xg, _pg = qflex_curve
            _fig.add_trace(go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name=f"{_qflex_display} fit",
                                        line=dict(color=_qflex_color, width=2.4)), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_xg, y=_pg, mode="lines", name=f"{_qflex_display} fit",
                                        line=dict(color=_qflex_color, width=2.4), showlegend=False), row=1, col=2)
            _n_modes, _locs, _hgts = qflex_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(go.Scatter(x=_locs, y=_hgts, mode="markers", name=f"{_qflex_display} modes",
                                            marker=dict(color=_qflex_color, size=10, symbol="diamond",
                                                        line=dict(color="white", width=1))), row=1, col=2)

        # Classical parametric baselines (normal / GEV), dashed so they read
        # as reference models rather than as another fitted QPD.
        for _ref in (reference_fits or []):
            _rx, _rp = _ref["curve"]
            _fig.add_trace(go.Scatter(x=FIT_P_GRID, y=_rx, mode="lines", name=f"{_ref['name']} fit",
                                        line=dict(color=_ref["color"], width=2.2, dash="dash")), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_rx, y=_rp, mode="lines", name=f"{_ref['name']} fit",
                                        line=dict(color=_ref["color"], width=2.2, dash="dash"),
                                        showlegend=False), row=1, col=2)

        # Guard the density axis against a runaway infeasible-fit spike -- a
        # known QFlex failure mode where an infeasible fit's PDF is 50-100x
        # taller than anything sensible (e.g. the Old Faithful waiting-time
        # data at default K). Left to Plotly's autorange, that single spike
        # stretches the y-axis so far that the histogram and every other
        # curve flatten to what looks like an empty plot. Anchor the range
        # on the histogram and any *feasible* fit instead; only fall back to
        # the raw curve max (which may itself be the spike) when nothing
        # feasible is available to anchor on, so the panel is never left
        # with no range at all.
        _hist_counts, _ = np.histogram(
            x_raw,
            bins=(np.arange(float(np.min(x_raw)) - bin_width / 2.0,
                            float(np.max(x_raw)) + bin_width, bin_width)
                  if bin_width else 40),
            density=True)
        _hist_max = float(np.max(_hist_counts)) if len(_hist_counts) else 0.0
        _feasible_maxes = []
        if metalog_fit is not None and getattr(metalog_fit, "is_feasible", False) and metalog_curve is not None:
            _feasible_maxes.append(float(np.max(metalog_curve[1])))
        if qflex_fit is not None and getattr(qflex_fit, "is_feasible", False) and qflex_curve is not None:
            _feasible_maxes.append(float(np.max(qflex_curve[1])))
        # Reference fits are parametric and always well-behaved, so they can
        # anchor the density axis unconditionally.
        for _ref in (reference_fits or []):
            _feasible_maxes.append(float(np.max(_ref["curve"][1])))

        if _hist_max > 0 or _feasible_maxes:
            _pdf_y_max = max([_hist_max] + _feasible_maxes) * 1.25
        else:
            _fallback_maxes = [float(np.max(_c[1])) for _c in (metalog_curve, qflex_curve) if _c is not None]
            _pdf_y_max = (max(_fallback_maxes) * 1.1) if _fallback_maxes else 1.0

        _fig.update_xaxes(title_text="Cumulative probability", range=[0, 1], row=1, col=1)
        _fig.update_yaxes(title_text=axis_label, row=1, col=1)
        _fig.update_xaxes(title_text=axis_label, row=1, col=2)
        _fig.update_yaxes(title_text="Density", range=[0, _pdf_y_max], row=1, col=2)

        # Optional fixed lower/upper bound on the value axis (the QF plot's
        # y-axis and the PDF plot's x-axis both show the same data domain,
        # e.g. weight or waiting time). Either side of value_xlim may be
        # None, in which case that side keeps a data-driven bound matching
        # Plotly's own default ~ padding rather than clipping to it exactly.
        if value_xlim is not None:
            _lo_override, _hi_override = value_xlim
            _value_arrays = [x_raw]
            if metalog_curve is not None:
                _value_arrays.append(metalog_curve[0])
            if qflex_curve is not None:
                _value_arrays.append(qflex_curve[0])
            for _ref in (reference_fits or []):
                _value_arrays.append(_ref["curve"][0])
            _value_all = np.concatenate(_value_arrays)
            _v_min, _v_max = float(np.min(_value_all)), float(np.max(_value_all))
            _v_pad = (_v_max - _v_min) * 0.05 or 1.0
            _v_lo = _lo_override if _lo_override is not None else (_v_min - _v_pad)
            _v_hi = _hi_override if _hi_override is not None else (_v_max + _v_pad)
            _fig.update_yaxes(range=[_v_lo, _v_hi], row=1, col=1)
            _fig.update_xaxes(range=[_v_lo, _v_hi], row=1, col=2)

        _fig.update_layout(
            height=460, margin=dict(l=55, r=25, t=45, b=75),
            legend=dict(orientation="h", y=-0.22), dragmode="zoom", barmode="overlay",
        )
        style_fig(_fig)

        # W1 (mean absolute quantile deviation) between each fit's quantile
        # curve and the empirical quantile function itself -- a per-fit
        # goodness-of-fit number in the same units/spirit as the "W1 vs
        # full-sample fit" column reported by the bootstrap batch below,
        # but here comparing the single current fit against the raw data
        # rather than against a batch of resampled refits.
        # Equation 6, against the observed sample's EQF.
        _eqf_on_w1 = np.interp(W1_P_GRID, p_grid, eqf_point)
        _L = PAPER.interdecile(_eqf_on_w1)
        def _eq6(curve):
            if curve is None:
                return None
            _q = np.interp(W1_P_GRID, FIT_P_GRID, curve[0])
            return PAPER.w1(_q, _eqf_on_w1, _L)[1]
        _metalog_w1 = _eq6(metalog_curve)
        _qflex_w1 = _eq6(qflex_curve)

        # Same W1 measure for each parametric baseline, on the same grid, so
        # the reference model and the QPDs are directly comparable.
        _ref_lines = []
        for _ref in (reference_fits or []):
            _rw1 = PAPER.w1(np.interp(W1_P_GRID, FIT_P_GRID, _ref["curve"][0]),
                            _eqf_on_w1, _L)[1]
            _rn = _ref.get("n_modes")
            _rshape = f", {_rn} mode{'s' if _rn and _rn > 1 else ''}" if _rn else ""
            _ref_lines.append(
                f"**{_ref['name']}:** reference fit{_rshape} | **W1 vs empirical** = {_rw1:.4f}"
            )

        _summary_block = mode_summary_md(
            mo, fit_error, metalog_fit, metalog_modes, qflex_fit, qflex_modes, constraint_label,
            metalog_w1=_metalog_w1, qflex_w1=_qflex_w1, bounds=bounds, true_n_modes=true_n_modes,
        )
        _parts = [_summary_block]
        if _ref_lines:
            _parts.append(mo.md("  \n".join(_ref_lines)))
        _parts.append(mo.ui.plotly(_fig, config=plotly_config))
        return mo.vstack(_parts)

    #: Units for the mode-IQR columns, matching the paper's own headers.
    UNITS = {"hydrology": ("ft", "1/ft"), "fish": ("lb", "1/lb"),
             "geyser": ("min", "1/min"), "bootstrap": ("", "")}

    #: What each section's table corresponds to in the manuscript.
    TABLE_CAPTIONS = {
        "mc": "**Tables 3 and A1** &mdash; validity, false modality and median W1, "
              "reported here for all four QPDs rather than the paper's two.",
        "bootstrap": "**Table 5** &mdash; validity, mode-dispersion IQRs, median W1 and the "
                     "mode-count split. (The Hartigan rate below is Table 4.)",
        "bimodal": "**Table 6** &mdash; mode recovery against a genuinely bimodal population, "
                   "so 2 modes is the correct answer and *>2 Modes* is the overfitting column.",
        "hydrology": "**Table 7** &mdash; the primary mode is the TALLEST peak of each fit.",
        "fish": "**Table 9** &mdash; the primary mode is the TALLEST peak, the secondary the "
                "next tallest.",
        "geyser": "**Table 10** &mdash; the primary mode is the one at the LONGER waiting time, "
                  "as the manuscript defines it, not the taller peak.",
    }

    def run_replicate_batch(mo, n_reps, k_metalog_val, k_qflex_val, draw_fn, seed, w1_ref=None,
                              w1_label="W1 vs reference", bounds=None, true_n_modes=1,
                              table_format="mc"):
        """Run a batch of replicates, fitting all 4 QPDs (Metalog + all 3
        QFlex constraint variants) to each one, then leave a summary behind
        -- feasibility rate, false-modality rate, and (when a reference is
        given) median W1 distance per model, the same statistics behind the
        paper's Tables 3 and A1, now covering all 4 QPDs rather than just the 2
        shown in the live panel above. Also runs the Hartigan dip test on
        each replicate's raw draw and reports the aggregate unimodality
        rejection rate. Only the final summary is shown -- not a
        per-iteration table -- since for 30-100 replicates x 4 models that
        table was mostly noise nobody read.

        w1_ref: one of
          * the string "own-sample" -- each replicate is measured against its
            OWN sample's EQF, rebuilt per replicate. This is what the paper's
            Monte Carlo tables do;
          * a single quantile-grid array on FIT_P_GRID, used for all 4 models
            (e.g. an empirical section's fixed observed EQF);
          * a dict {model_label: array_or_None} for a per-model reference
            (e.g. each section's own full-sample fit per model).
        Whichever form is given, the reference reaching the W1 computation is
        always an array on W1_P_GRID, so Equation 6 is evaluated on the paper's
        interval and never on the wider drawing grid."""
        _constraints = {"QFlex-U": "NONE", "QFlex-TA+": "TA", "QFlex-A+": "A"}
        # Whether draw_fn wants the replicate index is decided by INSPECTING it,
        # not by catching TypeError: a genuine TypeError raised inside a correct
        # draw_fn used to be swallowed and silently demote the section to the
        # legacy single-stream RNG, losing per-replicate seeding without a word.
        _draw_takes_rep = (getattr(draw_fn, "__code__", None) is not None
                           and draw_fn.__code__.co_argcount >= 2)
        _rng = np.random.default_rng(seed)
        _rows = []
        _dip_rejects = 0

        _OWN_SAMPLE = "own-sample"

        def _ref_for(label, own_ref):
            """Always returns a reference on W1_P_GRID (or None)."""
            if w1_ref is None:
                return None
            if isinstance(w1_ref, str):
                return own_ref
            _a = w1_ref.get(label) if isinstance(w1_ref, dict) else w1_ref
            return None if _a is None else np.interp(W1_P_GRID, FIT_P_GRID, _a)

        for _rep in range(n_reps):
            _x, _y = draw_fn(_rng, _rep) if _draw_takes_rep else draw_fn(_rng)
            # Built once per replicate, not once per model.
            _own_ref = (PAPER.empirical_qf(np.sort(_x), W1_P_GRID)
                        if isinstance(w1_ref, str) and w1_ref == _OWN_SAMPLE else None)

            try:
                _, _dip_pval, _dip_reject = hartigan_test(_x)
                _dip_rejects += int(_dip_reject)
            except Exception:
                pass

            for _model_name in MODEL_ORDER:
                _row = {"Replicate": _rep + 1, "Model": _model_name}
                _w1_ref_arr = _ref_for(_model_name, _own_ref)
                try:
                    if _model_name == "Metalog":
                        _fit = _make_metalog(_x, _y, k_metalog_val, bounds)
                    else:
                        _fit = _make_qflex(_x, _y, k_qflex_val, ConstraintType[_constraints[_model_name]], bounds)
                    # Modes and W1 are BOTH evaluated on the paper's grid,
                    # [0.01, 0.99]. FIT_P_GRID is the wide DRAWING grid; the
                    # extreme tails it reaches are exactly where spurious bumps
                    # appear, so counting modes out there reads high against
                    # the paper, whose detect_modes_in_pdf uses [0.01, 0.99].
                    _xw = np.asarray(_fit.quantile(W1_P_GRID), float)
                    # CLIPPED AT ZERO, as every reproduction script does before
                    # calling detect_modes_from_arrays. A negative trough left
                    # in place inflates a neighbouring peak's prominence, so an
                    # unclipped PDF can report more modes than the paper. On the
                    # paper's own data it never bites -- no fit tested, feasible
                    # or not, dips below zero on [0.01, 0.99] -- but the two
                    # would disagree silently the first time one did.
                    _pw = np.clip(np.asarray(_fit.pdf(W1_P_GRID), float), 0, None)
                    _n_modes, _m_locs, _m_hgts = detect_modes_from_arrays(_xw, _pw)
                    _row["Valid"] = bool(_fit.is_feasible)
                    # -1 marks an unusable PDF. The paper drops those from the
                    # modality denominator rather than scoring them as 0 modes.
                    _row["Modes"] = int(_n_modes) if _n_modes is not None else -1
                    # Mode positions feed the paper's mode-IQR columns. Ranking
                    # happens later, per section, because the rule differs:
                    # tallest peak for fish and hydrology, longest wait for the
                    # geyser (see paper_defaults).
                    _row["_locs"] = _m_locs
                    _row["_hgts"] = _m_hgts
                    if _w1_ref_arr is not None:
                        # Equation 6: mean |Q_F - Q_T| over [0.01, 0.99],
                        # divided by the TARGET's interdecile range. The old
                        # sum(|.|)*dp on FIT_P_GRID was neither normalised nor
                        # on the right interval.
                        _row[w1_label] = round(
                            PAPER.w1(_xw, _w1_ref_arr,
                                     PAPER.interdecile(_w1_ref_arr))[1], 4)
                except (MetalogError, QFlexError):
                    _row["Valid"] = False
                    _row["Modes"] = None
                    if _w1_ref_arr is not None:
                        _row[w1_label] = None
                _rows.append(_row)

            if _rep == 0 or (_rep + 1) % 5 == 0 or _rep == n_reps - 1:
                mo.output.replace(mo.md(f"Running replicate {_rep + 1} / {n_reps} (4 QPDs each)..."))

        _df = pd.DataFrame(_rows)

        # TABLE FORMAT PER SECTION. Each section reports the columns of the
        # paper table it corresponds to, not one generic shape:
        #
        #   "mc"        Tables 3 / A1  Validity %, False Modality %, Median W1
        #   "bootstrap" Table 5        + mode IQRs and the 1/2/>2 split
        #   "bimodal"   Table 6        Max # Modes, Valid, Unimodal, Bimodal, >2
        #   "hydrology" Table 7        1/2/>2 split + PRIMARY mode IQRs (ft)
        #   "fish"      Table 9        1/2 split + primary AND secondary IQRs (lb)
        #   "geyser"    Table 10       1/2/>2 split + primary and secondary (min)
        #
        # The paper lays Tables 3 / A1 out WIDE (two models side by side) to fit
        # the page; the notebook reports four QPDs, so the same columns are given
        # one model per row. Contents are the paper's, the layout is long.
        _fmt = table_format or "mc"
        _rank = {"geyser": PAPER.rank_modes_by_position}.get(_fmt, PAPER.rank_modes_by_height)

        def _mode_iqrs(g_valid):
            """(prim_loc, prim_ht, sec_loc, sec_ht, n_primary, n_secondary).

            Primary over valid fits with >= 1 mode, secondary over >= 2 -- the
            paper's denominators. Ranking is the section's own rule."""
            _pl, _ph, _sl, _sh = [], [], [], []
            for _lc, _ht, _nm in zip(g_valid["_locs"], g_valid["_hgts"], g_valid["Modes"]):
                if not (isinstance(_nm, (int, np.integer)) and _nm >= 1):
                    continue
                _a, _b, _c, _d = _rank(_lc, _ht)
                if not np.isfinite(_a):
                    continue
                _pl.append(_a); _ph.append(_b)
                if np.isfinite(_c):
                    _sl.append(_c); _sh.append(_d)
            return (PAPER.iqr(_pl), PAPER.iqr(_ph), PAPER.iqr(_sl), PAPER.iqr(_sh),
                    len(_pl), len(_sl))

        def _r(v, nd=4):
            return round(float(v), nd) if np.isfinite(v) else float("nan")

        _unit = UNITS.get(_fmt, ("", ""))     # (location unit, height unit)
        _summary_rows, _denoms = [], []
        for _model in MODEL_ORDER:
            _g = _df[_df["Model"] == _model]
            _n = len(_g)
            _feas = _g[_g["Valid"]]
            _feas_pct = round(100 * len(_feas) / _n, 1) if _n else 0.0
            # An unusable PDF (n_modes < 0) is dropped from the modality
            # denominator rather than scored as zero modes -- the paper's rule.
            _usable = _feas[_feas["Modes"] >= 0]
            _p1, _p2, _pg2 = PAPER.modality_split(_usable["Modes"].to_numpy())
            _thr = 1 if true_n_modes is None else true_n_modes
            _false_modal = (round(100 * float((_usable["Modes"] > _thr).mean()), 1)
                            if len(_usable) else 0.0)
            _k_used = k_metalog_val if _model == "Metalog" else k_qflex_val
            _prefix = _qpd_prefix(bounds)
            _w1 = (round(_feas[w1_label].median(), 4)
                   if (w1_label in _g.columns and len(_feas)) else float("nan"))
            _max_modes = int(_usable["Modes"].max()) if len(_usable) else 0

            if _fmt in ("hydrology", "fish"):
                # These two name the model WITH its order, as the paper does
                # ("Log Metalog-10"), and carry no separate K column.
                _name = f"{_prefix}{_model}-{_k_used}"
            else:
                _name = f"{_prefix}{_model}"

            if _fmt == "mc":
                _row_out = {"Model": _name, "K": _k_used, "Validity %": _feas_pct,
                            "False Modality %": round(_false_modal, 1),
                            "Median W1": _w1}
            elif _fmt == "bimodal":
                _row_out = {"QPD": _name, "Order (K)": _k_used,
                            "Max # Modes": _max_modes, "Valid (%)": _feas_pct,
                            "Unimodal (%)": round(_p1, 1), "Bimodal (%)": round(_p2, 1),
                            ">2 Modes (%)": round(_pg2, 1),
                            "Median W1 Distance": _w1}
            else:
                _il, _ih, _sl_, _sh_, _np_, _ns_ = _mode_iqrs(_feas)
                _denoms.append(f"{_name}: {len(_feas)} valid, {_np_} with a mode, {_ns_} with two")
                if _fmt == "bootstrap":
                    # Named, not just "W1 (Median)": this section measures each
                    # replicate against the reference sample it was resampled
                    # from, and the heading should say so.
                    _row_out = {"Model": _name, "K": _k_used, "Validity %": _feas_pct,
                                "Mode location IQR": _r(_il), "Mode height IQR": _r(_ih),
                                f"Median {w1_label}": _w1,
                                "1 mode %": round(_p1, 1), "2 modes %": round(_p2, 1),
                                ">2 modes %": round(_pg2, 1)}
                elif _fmt == "hydrology":
                    _row_out = {"Model": _name, "Validity %": _feas_pct,
                                "1 mode %": round(_p1, 1), "2 modes %": round(_p2, 1),
                                ">2 modes %": round(_pg2, 1),
                                f"Primary Mode Location ({_unit[0]})": _r(_il),
                                f"Primary Mode Height ({_unit[1]})": _r(_ih),
                                "W1 (Median)": _w1}
                elif _fmt == "fish":
                    _row_out = {"Model": _name, "Validity %": _feas_pct,
                                "1 mode %": round(_p1, 1), "2 modes %": round(_p2, 1),
                                f"Primary Location ({_unit[0]})": _r(_il),
                                f"Primary Height ({_unit[1]})": _r(_ih),
                                f"Secondary Location ({_unit[0]})": _r(_sl_),
                                f"Secondary Height ({_unit[1]})": _r(_sh_),
                                "Median W1": _w1}
                else:                                    # geyser
                    _row_out = {"Model": _name, "K": _k_used, "Validity %": _feas_pct,
                                "1 mode %": round(_p1, 1), "2 modes %": round(_p2, 1),
                                ">2 modes %": round(_pg2, 1),
                                f"Primary Location ({_unit[0]})": _r(_il),
                                f"Primary Height ({_unit[1]})": _r(_ih),
                                f"Secondary Location ({_unit[0]})": _r(_sl_),
                                f"Secondary Height ({_unit[1]})": _r(_sh_),
                                "W1 Median": _w1}
            _summary_rows.append(_row_out)

        _caption = TABLE_CAPTIONS.get(_fmt, "")
        # Denominators live under the table rather than in it, so the columns
        # stay the paper's. Validity % is out of all replicates; every other
        # column is conditioned on the valid fits, and the mode IQRs on the
        # narrower sets named here.
        _denom_md = ("  \n*Denominators:* " + "; ".join(_denoms) + f". Replicates: {n_reps}."
                     if _denoms else f"  \n*Validity % is out of {n_reps} replicates; "
                                     "every other column is conditioned on the valid fits.*")

        _dip_rate_pct = round(100 * _dip_rejects / n_reps, 1) if n_reps else 0.0

        mo.output.replace(
            mo.vstack([
                mo.md(f"**Done — {n_reps} replicates × 4 QPDs.**"),
                mo.md("**Summary across all replicates** &mdash; " + _caption + _denom_md),
                mo.ui.table(_summary_rows, selection=None, show_download=False, pagination=False),
                mo.md(
                    f"**Hartigan unimodality rejection rate:** {_dip_rate_pct}% of the {n_reps} replicate "
                    "draws reject unimodality at α=0.05 (dip test on the raw values themselves, independent "
                    "of any QPD fit)."
                ),
            ])
        )
        return

    return (
        FIT_P_GRID,
        MODEL_ORDER,
        QFLEX_LABELS,
        fit_all_qpds,
        fit_metalog_qflex,
        hartigan_line_md,
        hartigan_test,
        make_gev_reference,
        make_normal_reference,
        mode_summary_md,
        paper_settings_note,
        render_empirical_panel,
        render_mc_panel,
        run_replicate_batch,
        section_header_html,
        style_fig,
        true_dist_ranges,
    )


@app.cell
def _():
    # Trimmed Plotly modebar: keep pan/zoom (drag-to-zoom a region, drag to
    # pan) plus reset-axes and download, and drop the rest (box/lasso
    # select, the separate zoom-in/out step buttons, spike-line and
    # hover-compare toggles) since those are the icons nobody could place.
    PLOTLY_CONFIG = {
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
            "hoverClosestCartesian", "hoverCompareCartesian", "toggleSpikelines",
        ],
    }
    return (PLOTLY_CONFIG,)


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("Johnson distributions", level=2)}

        The reference ("true") population every Monte Carlo panel below
        draws from is a member of the Johnson system: one $(\eta, \kappa,
        c, d)$ recipe, expressed through three different quantile-function
        shapes (SU unbounded, SL semi-bounded, SB bounded on $(\eta, \eta+
        \kappa)$). Defaults match Bickel (2026) Figure 1. Vary the sliders
        to see how the same recipe reshapes each family.
        """
    )
    return


@app.cell
def _(mo):
    eta_j = mo.ui.slider(start=-3.0, stop=3.0, step=0.02, value=0.0, label="η (location)", show_value=True)
    kappa_j = mo.ui.slider(start=0.1, stop=4.0, step=0.02, value=1.0, label="κ (scale / SB width)", show_value=True)
    c_j = mo.ui.slider(start=-3.0, stop=3.0, step=0.02, value=0.5, label="c (shape)", show_value=True)
    d_j = mo.ui.slider(start=0.1, stop=4.0, step=0.02, value=1.2, label="d (shape/scale)", show_value=True)
    return c_j, d_j, eta_j, kappa_j


@app.cell
def _(c_j, d_j, eta_j, kappa_j, mo):
    mo.hstack([eta_j, kappa_j, c_j, d_j], justify="start", gap=2)
    return


@app.cell
def _(JohnsonSB, JohnsonSL, JohnsonSU, c_j, d_j, eta_j, kappa_j):
    su_dist = JohnsonSU(eta=eta_j.value, kappa=kappa_j.value, c=c_j.value, d=d_j.value)
    sl_dist = JohnsonSL(eta=eta_j.value, kappa=kappa_j.value, c=c_j.value, d=d_j.value)
    sb_dist = JohnsonSB(eta=eta_j.value, kappa=kappa_j.value, c=c_j.value, d=d_j.value)
    return sb_dist, sl_dist, su_dist


@app.cell
def _(PLOTLY_CONFIG, go, make_subplots, mo, np, sb_dist, sl_dist, style_fig, su_dist):
    _fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Johnson SU (unbounded)", "Johnson SL (semi-bounded)", "Johnson SB (bounded)"),
        horizontal_spacing=0.08,
    )

    _p = np.linspace(0.002, 0.998, 500)
    # figure_style.REFERENCE in the reproduction -- the three reference
    # distributions get colours that are NOT any model's. The previous trio
    # (#3B5FA0 / #2E8B57 / #A66A16) sat right next to Metalog, QFlex-A+ and
    # QFlex-TA+ respectively, which is the collision the paper's
    # reference-distributions figure was rebuilt to remove. (Open item, paper side: REFERENCE[0] #d62728 is also the KDE and
    # zoom-frame colour, so that red does double duty.)
    _specs = [(su_dist, "#d62728", 1), (sl_dist, "#17becf", 2), (sb_dist, "#bcbd22", 3)]
    for _dist, _color, _col in _specs:
        _x = _dist.quantile(_p)
        _pdf = _dist.pdf(_x)
        _fig.add_trace(
            go.Scatter(x=_x, y=_pdf, mode="lines", line=dict(color=_color, width=2.4), showlegend=False),
            row=1, col=_col,
        )
        _fig.update_xaxes(title_text="Value", row=1, col=_col)
        if _col == 1:
            _fig.update_yaxes(title_text="Density", row=1, col=_col)

    _fig.update_layout(height=360, margin=dict(l=55, r=20, t=45, b=50), showlegend=False)
    style_fig(_fig, dense_ticks=True)
    mo.ui.plotly(_fig, config=PLOTLY_CONFIG)
    return


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("Refitting under resampling", level=2)}

        The mechanism the whole paper is about, made tangible: draw a
        sample, fit Metalog and QFlex to it, and watch what the fits claim
        about the population. Two **independent experiments** follow, each
        with its own complete set of scenario controls, so you can either
        match their settings and compare like for like, or set them
        differently and probe one without disturbing the other:

        - **A. Monte Carlo** &mdash; every draw is a fresh, independent
          sample from the true distribution. This isolates *sampling
          variability*: how much a fit moves when you genuinely re-observe
          the population.
        - **B. Bootstrap** &mdash; one realization is fixed, and every draw
          resamples *that* sample with replacement. This is the situation
          you are actually in with real data, where the population is out
          of reach and one sample is all you have.

        Axes in each experiment are fixed to its own true-distribution
        range, so a K or N change never rescales the plot out from under
        you.

        **Note:** every section on this page assigns quantile probabilities
        to sorted samples using the paper's own Weibull plotting positions,
        $p_i = i/(n+1)$ (its Equation 3) &mdash; defined here for the Monte
        Carlo simulation study and reused, unchanged, for the empirical
        case studies further down the page.
        """
    )
    return


@app.cell
def _(mo):
    def make_scenario_controls(mo_ref, family_label):
        """Build one complete, independent set of scenario controls
        (reference family, N, both K's, QFlex constraint). Each experiment
        below owns its own set rather than sharing one block, so changing
        the Monte Carlo scenario never silently moves the bootstrap
        experiment's settings underneath it -- and the two can be set
        differently on purpose."""
        _family = mo_ref.ui.dropdown(
            options=[
                "Johnson SU (unbounded)",
                "Johnson SL (semi-bounded)",
                "Johnson SB (bounded)",
            ],
            value="Johnson SU (unbounded)",
            label=family_label,
        )
        _n = mo_ref.ui.slider(
            start=15, stop=500, step=5, value=200, label="Sample size N", show_value=True
        )
        # K = 10 is the paper's headline order: where Metalog starts
        # producing spurious modes and the constrained QFlex does not.
        _km = mo_ref.ui.slider(
            start=2, stop=15, step=1, value=10, label="Metalog K", show_value=True
        )
        _kq = mo_ref.ui.slider(
            start=2, stop=15, step=1, value=10, label="QFlex K", show_value=True
        )
        _cons = mo_ref.ui.dropdown(
            options={
                "Unconstrained": "NONE",
                "A+  (all coefficients ≥ 0)": "A",
                "TA+  (tail coefficients ≥ 0)": "TA",
            },
            # A+ is the comparator in the paper's unimodal Monte Carlo and
            # bootstrap sections -- its mode-count-and-dispersion figures put
            # Metalog against QFlex-A+.
            value="A+  (all coefficients \u2265 0)",
            label="QFlex constraint",
        )
        return _family, _n, _km, _kq, _cons

    def render_scenario_controls(mo_ref, family_ui, n_ui, heading):
        """Only the *data* scenario -- which population, how many points.
        The QPD order/constraint controls are rendered separately by
        render_model_controls, immediately above the plot they change, so
        you are not reaching back up past the draw button and the dip test
        to move a K slider."""
        return mo_ref.vstack(
            [mo_ref.md(heading), mo_ref.hstack([family_ui, n_ui], justify="start", gap=2)],
            gap=1,
        )

    def render_model_controls(mo_ref, km_ui, kq_ui, cons_ui):
        """The QPD order sliders and the QFlex constraint, placed right
        above the panel they drive (matching the empirical sections' own
        layout) so the control and the curve it moves are in view together."""
        return mo_ref.vstack(
            [mo_ref.hstack([km_ui, kq_ui], justify="start", gap=2), cons_ui],
            gap=1,
        )

    def effective_n(mo_ref, n_ui, km_ui, kq_ui):
        """K-aware floor on N: a fit needs at least K distinct points, so N
        can never drop below the larger of the two chosen K's or the fit
        crashes instead of failing gracefully. Returns (n_effective,
        notice_or_None)."""
        _floor = max(km_ui.value, kq_ui.value, 8)
        _n_eff = max(n_ui.value, _floor)
        _notice = (
            mo_ref.callout(
                f"N={n_ui.value} is below the current max K ({_floor}); "
                f"using N={_n_eff} so both fits stay solvable.",
                kind="warn",
            )
            if n_ui.value < _floor
            else None
        )
        return _n_eff, _notice

    def johnson_bounds(family_value, eta_value, kappa_value):
        """Match the boundedness of the fitted Metalog/QFlex to the support
        of whichever Johnson family is selected as the true population --
        unbounded (SU) stays unbounded, semi-bounded (SL, domain
        (eta, inf)) fits Log Metalog/Log QFlex with lower_bound=eta, and
        bounded (SB, domain (eta, eta+kappa)) fits Logit Metalog/Logit
        QFlex with both bounds. Mirrors the paper's own MC design, which
        obtains "the semi-bounded and bounded distributions via exponential
        and logit transforms, respectively" to match the reference family's
        support. Returns (bounds, note_markdown_text)."""
        if family_value.startswith("Johnson SU"):
            return None, "**Unbounded fit** — Johnson SU has support on the whole real line, so plain Metalog / QFlex are used."
        if family_value.startswith("Johnson SL"):
            return (eta_value, None), (
                f"**Semi-bounded fit (Log Metalog / Log QFlex)** — Johnson SL has support "
                f"(η, ∞) = ({eta_value:.2f}, ∞), so the lower bound is set to η = {eta_value:.2f}."
            )
        return (eta_value, eta_value + kappa_value), (
            f"**Bounded fit (Logit Metalog / Logit QFlex)** — Johnson SB has support "
            f"(η, η+κ) = ({eta_value:.2f}, {eta_value + kappa_value:.2f}), so both bounds are set accordingly."
        )

    def johnson_dist_for(family_value, su, sl, sb):
        return {
            "Johnson SU (unbounded)": su,
            "Johnson SL (semi-bounded)": sl,
            "Johnson SB (bounded)": sb,
        }[family_value]

    return effective_n, johnson_bounds, johnson_dist_for, make_scenario_controls, render_model_controls, render_scenario_controls


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("A. Monte Carlo — a fresh sample every time", level=3)}

        Each draw is a new, independent sample of N points from the true
        distribution. Because the population is genuinely re-observed each
        time, the spread you see across draws is pure **sampling
        variability** — the irreducible noise a fit has to cope with even
        when nothing about the data-generating process has changed.
        """
    )
    return


@app.cell
def _(mo, paper_settings_note):
    paper_settings_note(
        mo,
        "`42 + N×1000 + replication`, drawn with jpse's `rvs` &mdash; at N=200 that is `200042, 200043, …`. The paper's Monte Carlo uses this exact formula; drawing `quantile(rng.random(n))` instead gives a different sample from the same seed.",
        [("Reference distribution", "Johnson SU (η=0, κ=1, c=0.5, d=1.2)"), ("Sample size N", "200"), ("Metalog K", "10"), ("QFlex K", "10"), ("Replicates", "1000")],
        "Tables 3 and A1, and the Monte Carlo results figures (Johnson SU in the main "
        "text, SL and SB in the appendix). The paper reports K = 4, 7, 10 and 13; one run "
        "reproduces one of those blocks.",
    )
    return

@app.cell
def _(make_scenario_controls, mo):
    mc_family, mc_n_slider, mc_k_metalog, mc_k_qflex, mc_qflex_constraint = make_scenario_controls(
        mo, "Reference distribution (uses the η/κ/c/d panel above)"
    )
    return mc_family, mc_k_metalog, mc_k_qflex, mc_n_slider, mc_qflex_constraint


@app.cell
def _(mc_family, mc_n_slider, mo, render_scenario_controls):
    render_scenario_controls(
        mo, mc_family, mc_n_slider,
        "**Monte Carlo scenario controls** — independent of the bootstrap experiment below.",
    )
    return


@app.cell
def _(effective_n, mc_k_metalog, mc_k_qflex, mc_n_slider, mo):
    mc_n_effective, mc_n_notice = effective_n(mo, mc_n_slider, mc_k_metalog, mc_k_qflex)
    mc_n_notice
    return (mc_n_effective,)


@app.cell
def _(johnson_dist_for, mc_family, sb_dist, sl_dist, su_dist):
    mc_true_dist = johnson_dist_for(mc_family.value, su_dist, sl_dist, sb_dist)
    return (mc_true_dist,)


@app.cell
def _(eta_j, johnson_bounds, kappa_j, mc_family, mo):
    mc_bounds, _note = johnson_bounds(mc_family.value, eta_j.value, kappa_j.value)
    mo.md(_note)
    return (mc_bounds,)


@app.cell
def _(mc_true_dist, true_dist_ranges):
    mc_x_range, mc_y_range = true_dist_ranges(mc_true_dist)
    return mc_x_range, mc_y_range


@app.cell
def _(mo):
    mc_redraw = mo.ui.button(
        label="🎲 Draw a new Monte Carlo sample",
        value=0, on_click=lambda v: v + 1,
    )
    return (mc_redraw,)


@app.cell
def _(mc_redraw, mo):
    mo.vstack([
        mc_redraw,
        mo.md(f"*Draws so far: **#{mc_redraw.value + 1}** — each click refits on a brand-new sample from the true distribution.*"),
    ], gap=1)
    return


@app.cell
def _(PAPER, mc_base_seed, mc_n_effective, mc_redraw, mc_true_dist):
    # The live panel shows REPLICATE (redraw + 1) of the paper's own Monte
    # Carlo, so an untouched notebook displays the paper's first replicate and
    # the redraw button walks through 2, 3, ... Previously this drew from
    # `quantile(rng.random(n))` on an unrelated 10_000 + redraw stream, so the
    # panel above and the batch below were fitting different sample families
    # in the same section -- and neither matched the seed the callout states.
    mc_x_sample = PAPER.mc_draw(mc_true_dist, mc_n_effective,
                                mc_redraw.value + 1, base=mc_base_seed.value)
    _n = len(mc_x_sample)
    mc_y_sample = np.arange(1, _n + 1) / (_n + 1)  # Weibull plotting position, matching the paper's Equation 3
    return mc_x_sample, mc_y_sample


@app.cell
def _(hartigan_line_md, mc_x_sample, mo):
    hartigan_line_md(mo, mc_x_sample)
    return


@app.cell
def _(mc_k_metalog, mc_k_qflex, mc_qflex_constraint, mo, render_model_controls):
    render_model_controls(mo, mc_k_metalog, mc_k_qflex, mc_qflex_constraint)
    return


@app.cell
def _(fit_metalog_qflex, mc_bounds, mc_k_metalog, mc_k_qflex, mc_qflex_constraint, mc_x_sample, mc_y_sample):
    _res = fit_metalog_qflex(mc_x_sample, mc_y_sample, mc_k_metalog.value, mc_k_qflex.value, mc_qflex_constraint.value, bounds=mc_bounds)
    mc_fit_error = _res["fit_error"]
    mc_metalog_fit, mc_metalog_curve, mc_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    mc_qflex_fit, mc_qflex_curve, mc_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return mc_fit_error, mc_metalog_curve, mc_metalog_fit, mc_metalog_modes, mc_qflex_curve, mc_qflex_fit, mc_qflex_modes


@app.cell
def _(
    PLOTLY_CONFIG,
    mc_bounds,
    mc_fit_error,
    mc_k_metalog,
    mc_k_qflex,
    mc_metalog_curve,
    mc_metalog_fit,
    mc_metalog_modes,
    mc_qflex_constraint,
    mc_qflex_curve,
    mc_qflex_fit,
    mc_qflex_modes,
    mc_true_dist,
    mc_x_range,
    mc_x_sample,
    mc_y_range,
    mc_y_sample,
    mo,
    render_mc_panel,
):
    render_mc_panel(
        mo, PLOTLY_CONFIG, mc_true_dist, mc_x_sample, mc_y_sample, mc_k_metalog.value, mc_k_qflex.value,
        mc_qflex_constraint.value, mc_metalog_curve, mc_metalog_fit, mc_metalog_modes, mc_qflex_curve,
        mc_qflex_fit, mc_qflex_modes, mc_fit_error, mc_x_range, mc_y_range, bounds=mc_bounds,
    )
    return


@app.cell
def _(mo):
    mc_n_replicates = mo.ui.slider(
        start=5, stop=1000, step=5, value=PAPER.N_BOOT,
        label="Replicates", show_value=True
    )
    mc_run_batch = mo.ui.run_button(label="▶ Run Monte Carlo Analysis")
    # The draws are fully determined by (base, N, replicate), so pressing the
    # button twice gives identical output -- that is the point. Moving this
    # number draws a DIFFERENT but equally reproducible ensemble; 42 is the
    # paper's, and is what every published Monte Carlo number comes from.
    mc_base_seed = mo.ui.number(
        start=0, stop=999_999, step=1, value=PAPER.MC_BASE_SEED,
        label="Monte Carlo base seed (42 = the paper)",
    )
    return mc_base_seed, mc_n_replicates, mc_run_batch


@app.cell
def _(mc_base_seed, mc_n_replicates, mc_run_batch, mo):
    mo.vstack([
        mo.md(
            "**Full Monte Carlo simulation** — uses this experiment's own family, N, and K's; "
            "each replicate draws a brand-new sample from the true distribution. W1 is measured against "
            "the **true** quantile function, which is known here."
        ),
        mo.hstack([mc_n_replicates, mc_base_seed, mc_run_batch], justify="start", gap=2),
    ], gap=1)
    return


@app.cell
def _(
    FIT_P_GRID,
    mc_bounds,
    mc_k_metalog,
    mc_k_qflex,
    mc_n_effective,
    mc_n_replicates,
    mc_base_seed,
    mc_run_batch,
    mc_true_dist,
    mo,
    np,
    run_replicate_batch,
):
    if mc_run_batch.value:
        # The batch's W1 reference is each replicate's own sample EQF, so it is
        # rebuilt per replicate inside _draw below rather than fixed here.

        def _draw(rng, rep):
            # The paper's Monte Carlo draw: jpse's rvs (legacy np.random.seed
            # + inverse CDF) at seed = 42 + N*1000 + replication. Drawing
            # `dist.quantile(rng.random(n))` instead -- what this notebook did
            # before -- gives a DIFFERENT sample for the same nominal seed, so
            # the batch could never match the paper no matter how N and K were
            # set. Replication numbering starts at 1, as in the paper.
            _x = PAPER.mc_draw(mc_true_dist, mc_n_effective, rep + 1,
                               base=mc_base_seed.value)
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, mc_n_replicates.value, mc_k_metalog.value, mc_k_qflex.value,
            _draw, PAPER.mc_seed(mc_n_effective, 1, mc_base_seed.value),
            w1_ref="own-sample", w1_label="W1 vs sample", bounds=mc_bounds,
            table_format="mc"                      # Tables 3 / A1
        )
    else:
        mo.output.replace(
            mo.md("*Click **▶ Run Monte Carlo Analysis** to draw fresh samples repeatedly and summarize all 4 QPDs.*")
        )
    return


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("B. Bootstrap — one fixed sample, resampled", level=3)}

        Now the population is out of reach. **One** realization is drawn
        and held fixed (that is the "reference realization" below), and
        every replicate resamples *it* with replacement. This is the
        situation real data puts you in, and the spread you see is what a
        bootstrap can actually tell you about a fit's stability — measured
        against the reference sample rather than a truth you would not have.

        This experiment has its own controls: set them to match the Monte
        Carlo scenario above for a like-for-like comparison, or differently
        to explore the bootstrap on its own terms.
        """
    )
    return


@app.cell
def _(mo, paper_settings_note):
    paper_settings_note(
        mo,
        "reference realization `200043` (= `42 + 200×1000 + 1`, i.e. replication 1 of the Monte Carlo above); each resample uses `200043×10000 + b`, so replicate *b* is reachable on its own rather than by replaying a loop.",
        [("Reference-sample seed", "200043"), ("Sample size N", "200"), ("Metalog K", "10"), ("QFlex K", "10"), ("Replicates", "1000")],
        "Tables 4 and 5, and the bootstrap figures. The paper reports K = 4, 7, 10 and 13; "
        "persistence at K = 4 / 7 / 10 is 100 / 66.3 / 37.9 %.",
    )
    return

@app.cell
def _(make_scenario_controls, mo):
    boot_family, boot_n_slider, boot_k_metalog, boot_k_qflex, boot_qflex_constraint = make_scenario_controls(
        mo, "Reference distribution (uses the η/κ/c/d panel above)"
    )
    return boot_family, boot_k_metalog, boot_k_qflex, boot_n_slider, boot_qflex_constraint


@app.cell
def _(boot_family, boot_n_slider, mo, render_scenario_controls):
    render_scenario_controls(
        mo, boot_family, boot_n_slider,
        "**Bootstrap scenario controls** — independent of the Monte Carlo experiment above.",
    )
    return


@app.cell
def _(boot_k_metalog, boot_k_qflex, boot_n_slider, effective_n, mo):
    boot_n_effective, boot_n_notice = effective_n(mo, boot_n_slider, boot_k_metalog, boot_k_qflex)
    boot_n_notice
    return (boot_n_effective,)


@app.cell
def _(boot_family, johnson_dist_for, sb_dist, sl_dist, su_dist):
    boot_true_dist = johnson_dist_for(boot_family.value, su_dist, sl_dist, sb_dist)
    return (boot_true_dist,)


@app.cell
def _(boot_family, eta_j, johnson_bounds, kappa_j, mo):
    boot_bounds, _note = johnson_bounds(boot_family.value, eta_j.value, kappa_j.value)
    mo.md(_note)
    return (boot_bounds,)


@app.cell
def _(boot_true_dist, true_dist_ranges):
    boot_x_range, boot_y_range = true_dist_ranges(boot_true_dist)
    return boot_x_range, boot_y_range


@app.cell
def _(mo):
    base_seed = mo.ui.number(
        start=0, stop=999_999, step=1, value=PAPER.BOOTSTRAP_SEED,
        label="Reference-sample seed",
    )
    # ONE control decides WHICH realization is bootstrapped: the seed box.
    # A "↻ draw a new realization" button used to sit beside it doing the same
    # job by adding an invisible offset to that seed, so two controls moved the
    # same thing and the box stopped showing the seed actually in use. The dice
    # button stays: it does something different -- it holds the realization
    # fixed and draws another resample from it, which is the bootstrap step.
    boot_redraw = mo.ui.button(
        label="🎲 Draw another resample",
        value=0, on_click=lambda v: v + 1,
    )
    return base_seed, boot_redraw


@app.cell
def _(PAPER, base_seed, boot_redraw, mo):
    _is_paper = " (the paper's)" if int(base_seed.value) == PAPER.BOOTSTRAP_SEED else ""
    mo.vstack([
        mo.hstack([base_seed, boot_redraw], justify="start", gap=2),
        mo.md(
            f"*Reference realization: seed **{int(base_seed.value)}**{_is_paper}"
            f" &nbsp;&middot;&nbsp; resample **#{boot_redraw.value + 1}***  \n"
            "*The **seed** chooses **which** single sample is being bootstrapped — the one standing "
            "in for \"the data you happen to have\". **🎲** holds that sample fixed and draws another "
            "resample from it, which is the bootstrap step itself.*"
        ),
    ], gap=1)
    return


@app.cell
def _(base_seed, boot_n_effective, boot_true_dist, np):
    # The "single selected realization" the bootstrap is conditioned on.
    # Regenerated when the reference seed changes, the "new reference
    # realization" button is clicked, or N changes (so it always matches the
    # current sample-size setting).
    #
    # DRAWN THE PAPER'S WAY. This used to be
    # `np.sort(dist.quantile(rng.random(n)))`, which is precisely the generator
    # path paper_defaults documents as wrong: jpse's `rvs` goes through the
    # legacy np.random.seed + inverse-CDF route, so the two produce DIFFERENT
    # samples from the same nominal seed. The whole bootstrap section was
    # therefore conditioned on a realization the paper never used.
    #
    # Two forms are returned. `reference_sample_raw` is in GENERATION order and
    # is what resampling must use -- rng.choice selects by INDEX, so drawing
    # from the sorted copy silently changes which values come out.
    # `reference_sample` is the sorted copy, for the EQF and the plots.
    # The seed box IS the realization. Its default, 200043, is exactly
    # PAPER.mc_seed(200, 1) -- the paper's pinned realization -- so an untouched
    # notebook still reproduces the published numbers. Any other value is a
    # different realization drawn the same way, differing in seed, not generator.
    reference_sample_raw = np.asarray(
        boot_true_dist.rvs(size=boot_n_effective,
                           random_state=int(base_seed.value)), float)
    reference_sample = np.sort(reference_sample_raw)
    return reference_sample, reference_sample_raw


@app.cell
def _(PAPER, boot_redraw, np, reference_sample_raw):
    # One bootstrap resample of that fixed reference realization. Resampling is
    # from the GENERATION-order copy and uses the paper's per-replicate stream,
    # so "resample b" here is the same draw the paper calls b.
    boot_x_sample = np.sort(PAPER.bootstrap_resample(reference_sample_raw,
                                                     boot_redraw.value))
    _n = len(boot_x_sample)
    boot_y_sample = np.arange(1, _n + 1) / (_n + 1)
    return boot_x_sample, boot_y_sample


@app.cell
def _(boot_x_sample, hartigan_line_md, mo):
    hartigan_line_md(mo, boot_x_sample)
    return


@app.cell
def _(boot_k_metalog, boot_k_qflex, boot_qflex_constraint, mo, render_model_controls):
    render_model_controls(mo, boot_k_metalog, boot_k_qflex, boot_qflex_constraint)
    return


@app.cell
def _(boot_bounds, boot_k_metalog, boot_k_qflex, boot_qflex_constraint, boot_x_sample, boot_y_sample, fit_metalog_qflex):
    _res = fit_metalog_qflex(boot_x_sample, boot_y_sample, boot_k_metalog.value, boot_k_qflex.value, boot_qflex_constraint.value, bounds=boot_bounds)
    boot_fit_error = _res["fit_error"]
    boot_metalog_fit, boot_metalog_curve, boot_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    boot_qflex_fit, boot_qflex_curve, boot_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return boot_fit_error, boot_metalog_curve, boot_metalog_fit, boot_metalog_modes, boot_qflex_curve, boot_qflex_fit, boot_qflex_modes


@app.cell
def _(
    PLOTLY_CONFIG,
    boot_bounds,
    boot_fit_error,
    boot_k_metalog,
    boot_k_qflex,
    boot_metalog_curve,
    boot_metalog_fit,
    boot_metalog_modes,
    boot_qflex_constraint,
    boot_qflex_curve,
    boot_qflex_fit,
    boot_qflex_modes,
    boot_true_dist,
    boot_x_range,
    boot_x_sample,
    boot_y_range,
    boot_y_sample,
    mo,
    render_mc_panel,
):
    render_mc_panel(
        mo, PLOTLY_CONFIG, boot_true_dist, boot_x_sample, boot_y_sample, boot_k_metalog.value, boot_k_qflex.value,
        boot_qflex_constraint.value, boot_metalog_curve, boot_metalog_fit, boot_metalog_modes, boot_qflex_curve,
        boot_qflex_fit, boot_qflex_modes, boot_fit_error, boot_x_range, boot_y_range, bounds=boot_bounds,
    )
    return


@app.cell
def _(mo):
    boot_n_replicates = mo.ui.slider(
        start=5, stop=1000, step=5, value=PAPER.N_BOOT,
        label="Replicates", show_value=True
    )
    boot_run_batch = mo.ui.run_button(label="▶ Run Bootstrap Analysis")
    return boot_n_replicates, boot_run_batch


@app.cell
def _(boot_n_replicates, boot_run_batch, mo):
    mo.vstack([
        mo.md(
            "**Full bootstrap simulation** — uses this experiment's own family, N, and K's; "
            "each replicate resamples the fixed reference realization. W1 is measured against that "
            "**reference sample's own EQF**, since in a real bootstrap the truth is exactly what you "
            "do not have."
        ),
        mo.hstack([boot_n_replicates, boot_run_batch], justify="start", gap=2),
    ], gap=1)
    return


@app.cell
def _(
    FIT_P_GRID,
    base_seed,
    boot_bounds,
    boot_k_metalog,
    boot_k_qflex,
    boot_n_effective,
    boot_n_replicates,
    boot_run_batch,
    PAPER,
    mo,
    np,
    reference_sample,
    reference_sample_raw,
    run_replicate_batch,
):
    if boot_run_batch.value:
        # Reference for W1 is the fixed realization's own EQF, interpolated
        # onto the fit grid -- the bootstrap analog of "the truth you have".
        _n_ref = len(reference_sample)
        _p_ref = np.arange(1, _n_ref + 1) / (_n_ref + 1)
        _x_ref_grid = np.interp(FIT_P_GRID, _p_ref, reference_sample)

        def _draw(rng, rep):
            # Per-replicate seeding on the paper's stream
            # (BOOTSTRAP_SEED*10_000 + b), from the generation-order copy.
            # Previously this took only `rng`, so run_replicate_batch's
            # signature check fell back to one shared stream and replicate b
            # was not the paper's replicate b.
            _x = np.sort(PAPER.bootstrap_resample(reference_sample_raw, rep))
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, boot_n_replicates.value, boot_k_metalog.value, boot_k_qflex.value,
            _draw, base_seed.value + 777, w1_ref=_x_ref_grid,
            w1_label="W1 vs reference sample", bounds=boot_bounds,
            table_format="bootstrap"
        )
    else:
        mo.output.replace(
            mo.md("*Click **▶ Run Bootstrap Analysis** to resample the fixed reference repeatedly and summarize all 4 QPDs.*")
        )
    return




@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("Bimodal Johnson distribution", level=2)}

        A two-component mixture built from the **Johnson SU panel above**:
        both components share that panel's shape, offset from each other by
        a controllable distance, and combined with a controllable mixture
        weight. This is the paper's motivating case made explorable &mdash;
        watch Metalog/QFlex either recover the real second mode or invent
        one that isn't there. Both components are built from Johnson SU,
        which is unbounded, so plain (unbounded) Metalog / QFlex are used
        here throughout.

        As above, the Monte Carlo and bootstrap experiments are kept
        **separate**, each with its own full set of controls: **A** redraws
        the mixture from scratch each time, **B** fixes one realization and
        resamples it. Set the two to the same separation, weight, N and K
        to compare them like for like, or differently to probe one on its
        own.
        """
    )
    return


@app.cell
def _(
    PAPER,
    detect_modes_from_arrays,
    mo,
    np,
    su_dist,):
    def make_bimodal_controls():
        """One complete, independent set of controls for a bimodal
        experiment. Each of A and B owns its own set, so retuning the
        separation or K for one never moves the other underneath it."""
        # Defaults are the paper's bimodal reference: a 60/40 mixture of
        # two Johnson SU distributions with the second shifted 3.5 SDs left,
        # N = 200, and K = 12 -- the order at which the paper's bimodal
        # reference figure shows
        # QFlex-TA+ recovering both modes.
        # The slider is SIGNED and starts at the paper's -3.5. It used to
        # run 0..6 with value=abs(PAPER.BIMODAL_SEPARATION), which threw the
        # sign away and contradicted the comment three lines above: the
        # notebook opened on a mixture shifted RIGHT. Because the base SU is
        # skewed (c = 0.5), that is not the mirror image but a different
        # population, W1 = 0.589 from the paper's.
        # show_value is off on all three: marimo renders that number in a very
        # light grey beside the track, which is easy to miss on the one control
        # whose exact value decides the experiment. render_bimodal_controls
        # prints each value in body text instead, next to what it means.
        _delta = mo.ui.slider(
            start=-6.0, stop=6.0, step=0.1, value=PAPER.BIMODAL_SEPARATION,
            label="", show_value=False,
        )
        _ratio = mo.ui.slider(
            start=0.05, stop=0.95, step=0.05, value=PAPER.BIMODAL_WEIGHTS[0],
            label="", show_value=False,
        )
        _n = mo.ui.slider(
            start=15, stop=500, step=5, value=PAPER.BIMODAL_N,
            label="", show_value=False,
        )
        _km = mo.ui.slider(start=2, stop=15, step=1, value=12, label="Metalog K", show_value=True)
        _kq = mo.ui.slider(start=2, stop=15, step=1, value=12, label="QFlex K", show_value=True)
        _cons = mo.ui.dropdown(
            options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
            value="TA+  (tail coefficients ≥ 0)", label="QFlex constraint",
        )
        return _delta, _ratio, _n, _km, _kq, _cons

    def render_bimodal_controls(delta_ui, ratio_ui, n_ui, heading):
        """Mixture shape and sample size only -- the K sliders and QFlex
        constraint are rendered next to the plot instead (see
        render_model_controls).

        Each control gets its own row, a full-sentence label saying what it
        decides, and its current value in body text. The previous version put
        the separation and weight sliders side by side with terse labels -- one
        of them said only "share on component A", with nothing on the page
        saying what A was, how many components there were, or that the weights
        sum to one."""
        _sd = PAPER.BIMODAL_BASE_SIGMA
        _delta, _wa = float(delta_ui.value), float(ratio_ui.value)
        _side = "left" if _delta < 0 else ("right" if _delta > 0 else "on top of A")
        _shift = _delta * _sd
        return mo.vstack(
            [
                mo.md(heading),
                mo.md(
                    "The population is a mixture of **two Johnson SU components** of "
                    "identical shape (η=0, κ=1, c=0.5, d=1.2): **component A**, which "
                    "stays put, and **component B**, a copy of it shifted along the "
                    f"x-axis. One standard deviation of that shape is **{_sd:.4f}**, and "
                    "the two controls below set where B sits and how much of the "
                    "population it holds."
                ),
                mo.md(f"**Separation of component B from A** — currently **{_delta:+.1f} SD** "
                      f"(a shift of {_shift:+.4f} in x, i.e. to the **{_side}**). "
                      "Negative moves B left, positive right; at 0 the two coincide and "
                      "the population is unimodal."),
                delta_ui,
                mo.md(f"**Share of the population in component A** — currently "
                      f"**{_wa:.0%} A / {1 - _wa:.0%} B**. The two shares sum to 100 %, so "
                      "this single control sets both."),
                ratio_ui,
                mo.md(f"**Sample size N** — currently **{int(n_ui.value)}** points drawn per "
                      "replicate."),
                n_ui,
            ],
            gap=1,
        )

    class _ShiftedSU:
        """su_dist translated by a fixed offset -- same shape, new eta."""
        def __init__(self, base, offset):
            self._base = base
            self._offset = offset

        def quantile(self, p):
            return self._base.quantile(p) + self._offset

        def pdf(self, x):
            return self._base.pdf(np.asarray(x) - self._offset)

    class _MixtureDist:
        """Numeric PDF + quantile for the mixture, used only for the "true
        curve" overlay and the true-mode count -- actual sampling inverts
        each component's exact quantile function directly, not this
        numeric approximation."""
        def __init__(self, comp_a, comp_b, weight_a):
            _lo = min(comp_a.quantile(0.0005), comp_b.quantile(0.0005))
            _hi = max(comp_a.quantile(0.9995), comp_b.quantile(0.9995))
            _span = _hi - _lo
            self._x_grid = np.linspace(_lo - 0.05 * _span, _hi + 0.05 * _span, 4000)
            self._pdf_grid = (
                weight_a * comp_a.pdf(self._x_grid) + (1 - weight_a) * comp_b.pdf(self._x_grid)
            )
            _cdf = np.cumsum(self._pdf_grid)
            _dx = self._x_grid[1] - self._x_grid[0]
            _cdf = _cdf * _dx
            self._cdf_grid = _cdf / _cdf[-1]

        def pdf(self, x):
            return np.interp(x, self._x_grid, self._pdf_grid, left=0.0, right=0.0)

        def quantile(self, p):
            return np.interp(np.asarray(p), self._cdf_grid, self._x_grid)

    def build_mixture(delta_value, ratio_value):
        """Assemble one mixture scenario: component A is exactly the Johnson
        SU panel above, component B is the same shape shifted by
        delta * sigma (sigma estimated numerically off a fine quantile grid,
        since Johnson quantile functions have no closed-form variance).
        Also reports how many modes the mixture *actually* has at this
        separation, read off the true density rather than assumed -- at
        small separations the two components merge into one peak, so
        hard-coding 2 would mislabel a correct unimodal fit as having
        missed a mode."""
        # sigma is the CLOSED FORM (PAPER.base_sigma), not np.std of the
        # quantile function on a [0.0005, 0.9995] grid. That grid truncates
        # enough of this SU's tail to read 1.3244 against a true 1.3691 --
        # 3.26 % low, which moved the whole population.
        _sigma = PAPER.base_sigma(eta=su_dist.eta, kappa=su_dist.kappa,
                                  c=su_dist.c, d=su_dist.d)
        _a = su_dist
        _b = _ShiftedSU(su_dist, delta_value * _sigma)
        _dist = _MixtureDist(_a, _b, ratio_value)
        _p = np.linspace(0.0005, 0.9995, 2000)
        _xg = _dist.quantile(_p)
        _n_true, _, _ = detect_modes_from_arrays(_xg, _dist.pdf(_xg))
        return {
            "comp_a": _a, "comp_b": _b, "weight_a": ratio_value,
            "base": su_dist, "offset": delta_value * _sigma, "sigma": _sigma,
            "dist": _dist, "true_modes": int(_n_true) if _n_true else 1,
        }

    def draw_mixture(seed_base, scenario, n):
        """The paper's mixture sampler: one binomial for the component split,
        then each component drawn by jpse's `rvs` on its own offset seed.

        This used to pick a component per draw (`rng.random(n) < w`) and invert
        that component's quantile. Both are exact samplers of the same mixture,
        but they give DIFFERENT samples from the same seed, so the notebook's
        bimodal section could not reproduce the paper's. Verified bit-identical
        to reproduction/scripts/bimodal_common.draw for replicates 0, 1, 7,
        436, 607 and 999."""
        return PAPER.bimodal_draw(scenario["base"], n, int(seed_base),
                                  scenario["offset"], scenario["weight_a"])

    return build_mixture, draw_mixture, make_bimodal_controls, render_bimodal_controls


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("A. Monte Carlo — a fresh sample every time", level=3)}

        Each draw is a new, independent sample from the true mixture, taken
        by inverting each component's exact quantile function. With a real
        second mode present, the question here is whether a fit recovers it
        — and how much the recovered mode moves from draw to draw.
        """
    )
    return


@app.cell
def _(mo, paper_settings_note):
    paper_settings_note(
        mo,
        "`42 + 60,000,000 + replication`. Each replicate is an independent draw from the analytic mixture &mdash; the paper's Table 6 was re-run this way, replacing an earlier pooled bootstrap of subsamples.",
        [("Mixture", "60 / 40, second component shifted 3.5 SD left"), ("Sample size N", "200"), ("Metalog K", "12"), ("QFlex K", "12"), ("QFlex constraint", "TA+"), ("Replicates", "1000")],
        "Table 6 and the bimodal reference figure, which span K = 4 to 14.",
    )
    return

@app.cell
def _(make_bimodal_controls):
    (bimodal_mc_delta, bimodal_mc_ratio, bimodal_mc_n_slider,
     bimodal_mc_k_metalog, bimodal_mc_k_qflex, bimodal_mc_constraint) = make_bimodal_controls()
    return bimodal_mc_constraint, bimodal_mc_delta, bimodal_mc_k_metalog, bimodal_mc_k_qflex, bimodal_mc_n_slider, bimodal_mc_ratio


@app.cell
def _(
    bimodal_mc_delta,
    bimodal_mc_n_slider,
    bimodal_mc_ratio,
    render_bimodal_controls,
):
    render_bimodal_controls(
        bimodal_mc_delta, bimodal_mc_ratio, bimodal_mc_n_slider,
        "**Monte Carlo scenario controls** — independent of the bootstrap experiment below.",
    )
    return


@app.cell
def _(bimodal_mc_k_metalog, bimodal_mc_k_qflex, bimodal_mc_n_slider, effective_n, mo):
    bimodal_mc_n_effective, _notice = effective_n(mo, bimodal_mc_n_slider, bimodal_mc_k_metalog, bimodal_mc_k_qflex)
    _notice
    return (bimodal_mc_n_effective,)


@app.cell
def _(bimodal_mc_delta, bimodal_mc_ratio, build_mixture):
    bimodal_mc_scenario = build_mixture(bimodal_mc_delta.value, bimodal_mc_ratio.value)
    return (bimodal_mc_scenario,)


@app.cell
def _(bimodal_mc_scenario, true_dist_ranges):
    bimodal_mc_x_range, bimodal_mc_y_range = true_dist_ranges(
        bimodal_mc_scenario["dist"], p_lo=0.0005, p_hi=0.9995
    )
    return bimodal_mc_x_range, bimodal_mc_y_range


@app.cell
def _(mo):
    bimodal_redraw = mo.ui.button(label="🎲 Draw a new Monte Carlo sample", value=0, on_click=lambda v: v + 1)
    return (bimodal_redraw,)


@app.cell
def _(bimodal_redraw, mo):
    mo.vstack([
        bimodal_redraw,
        mo.md(f"*Draws so far: **#{bimodal_redraw.value + 1}** — each click refits on a brand-new sample from the true mixture.*"),
    ], gap=1)
    return


@app.cell
def _(PAPER, bimodal_mc_n_effective, bimodal_mc_scenario, bimodal_redraw, draw_mixture, np):
    # Replicate (redraw + 1) of the paper's bimodal Monte Carlo stream
    # (42 + 60,000,000 + replication), matching the batch below; this used to
    # be an unrelated 20_000 + redraw stream.
    bimodal_mc_x_sample = draw_mixture(
        PAPER.bimodal_seed(bimodal_redraw.value + 1),
        bimodal_mc_scenario, bimodal_mc_n_effective)
    _n = len(bimodal_mc_x_sample)
    bimodal_mc_y_sample = np.arange(1, _n + 1) / (_n + 1)
    return bimodal_mc_x_sample, bimodal_mc_y_sample


@app.cell
def _(bimodal_mc_x_sample, hartigan_line_md, mo):
    hartigan_line_md(mo, bimodal_mc_x_sample)
    return


@app.cell
def _(bimodal_mc_constraint, bimodal_mc_k_metalog, bimodal_mc_k_qflex, mo, render_model_controls):
    render_model_controls(mo, bimodal_mc_k_metalog, bimodal_mc_k_qflex, bimodal_mc_constraint)
    return


@app.cell
def _(
    bimodal_mc_constraint,
    bimodal_mc_k_metalog,
    bimodal_mc_k_qflex,
    bimodal_mc_x_sample,
    bimodal_mc_y_sample,
    fit_metalog_qflex,
):
    _res = fit_metalog_qflex(
        bimodal_mc_x_sample, bimodal_mc_y_sample, bimodal_mc_k_metalog.value,
        bimodal_mc_k_qflex.value, bimodal_mc_constraint.value,
    )
    bimodal_mc_fit_error = _res["fit_error"]
    bimodal_mc_metalog_fit, bimodal_mc_metalog_curve, bimodal_mc_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    bimodal_mc_qflex_fit, bimodal_mc_qflex_curve, bimodal_mc_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return bimodal_mc_fit_error, bimodal_mc_metalog_curve, bimodal_mc_metalog_fit, bimodal_mc_metalog_modes, bimodal_mc_qflex_curve, bimodal_mc_qflex_fit, bimodal_mc_qflex_modes


@app.cell
def _(
    PLOTLY_CONFIG,
    bimodal_mc_constraint,
    bimodal_mc_fit_error,
    bimodal_mc_k_metalog,
    bimodal_mc_k_qflex,
    bimodal_mc_metalog_curve,
    bimodal_mc_metalog_fit,
    bimodal_mc_metalog_modes,
    bimodal_mc_qflex_curve,
    bimodal_mc_qflex_fit,
    bimodal_mc_qflex_modes,
    bimodal_mc_scenario,
    bimodal_mc_x_range,
    bimodal_mc_x_sample,
    bimodal_mc_y_range,
    bimodal_mc_y_sample,
    mo,
    render_mc_panel,
):
    render_mc_panel(
        mo, PLOTLY_CONFIG, bimodal_mc_scenario["dist"], bimodal_mc_x_sample, bimodal_mc_y_sample,
        bimodal_mc_k_metalog.value, bimodal_mc_k_qflex.value, bimodal_mc_constraint.value,
        bimodal_mc_metalog_curve, bimodal_mc_metalog_fit, bimodal_mc_metalog_modes,
        bimodal_mc_qflex_curve, bimodal_mc_qflex_fit, bimodal_mc_qflex_modes,
        bimodal_mc_fit_error, bimodal_mc_x_range, bimodal_mc_y_range,
        true_n_modes=bimodal_mc_scenario["true_modes"],
    )
    return


@app.cell
def _(mo):
    bimodal_n_replicates = mo.ui.slider(start=5, stop=1000, step=5, value=PAPER.N_BOOT, label="Replicates", show_value=True)
    bimodal_run_batch = mo.ui.run_button(label="▶ Run Monte Carlo Analysis")
    return bimodal_n_replicates, bimodal_run_batch


@app.cell
def _(bimodal_n_replicates, bimodal_run_batch, mo):
    mo.vstack([
        mo.md(
            "**Full Monte Carlo simulation** — uses this experiment's own mixture, N, and K's; "
            "each replicate draws a brand-new sample from the true mixture. W1 is measured against the "
            "**true** quantile function."
        ),
        mo.hstack([bimodal_n_replicates, bimodal_run_batch], justify="start", gap=2),
    ], gap=1)
    return


@app.cell
def _(
    FIT_P_GRID,
    bimodal_mc_k_metalog,
    bimodal_mc_k_qflex,
    bimodal_mc_n_effective,
    bimodal_mc_scenario,
    bimodal_n_replicates,
    bimodal_run_batch,
    draw_mixture,
    mo,
    np,
    run_replicate_batch,
):
    if bimodal_run_batch.value:
        def _draw(rng, rep):
            # Per-replicate seeding on the paper's bimodal stream
            # (42 + 60,000,000 + replication), replacing a single arbitrary
            # 606_060 stream consumed in a loop.
            _x = draw_mixture(PAPER.bimodal_seed(rep + 1),
                              bimodal_mc_scenario, bimodal_mc_n_effective)
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, bimodal_n_replicates.value, bimodal_mc_k_metalog.value, bimodal_mc_k_qflex.value,
            _draw, PAPER.bimodal_seed(1), w1_ref="own-sample", w1_label="W1 vs sample",
            true_n_modes=bimodal_mc_scenario["true_modes"],
            table_format="bimodal"                 # Table 6
        )
    else:
        mo.output.replace(
            mo.md("*Click **▶ Run Monte Carlo Analysis** to draw fresh samples repeatedly and summarize all 4 QPDs.*")
        )
    return


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("B. Bootstrap — one fixed sample, resampled", level=3)}

        One realization of the mixture is drawn and held fixed, and every
        replicate resamples *it*. This is the honest version of the modality
        question: given a single finite sample, how stable is the second
        mode a fit reports — its existence, its location, and its height?

        This experiment has its own controls, so the mixture it bootstraps
        can be set independently of the Monte Carlo one above.
        """
    )
    return


@app.cell
def _(make_bimodal_controls):
    (bimodal_boot_delta, bimodal_boot_ratio, bimodal_boot_n_slider,
     bimodal_boot_k_metalog, bimodal_boot_k_qflex, bimodal_boot_constraint) = make_bimodal_controls()
    return bimodal_boot_constraint, bimodal_boot_delta, bimodal_boot_k_metalog, bimodal_boot_k_qflex, bimodal_boot_n_slider, bimodal_boot_ratio


@app.cell
def _(
    bimodal_boot_delta,
    bimodal_boot_n_slider,
    bimodal_boot_ratio,
    render_bimodal_controls,
):
    render_bimodal_controls(
        bimodal_boot_delta, bimodal_boot_ratio, bimodal_boot_n_slider,
        "**Bootstrap scenario controls** — independent of the Monte Carlo experiment above.",
    )
    return


@app.cell
def _(bimodal_boot_k_metalog, bimodal_boot_k_qflex, bimodal_boot_n_slider, effective_n, mo):
    bimodal_boot_n_effective, _notice = effective_n(mo, bimodal_boot_n_slider, bimodal_boot_k_metalog, bimodal_boot_k_qflex)
    _notice
    return (bimodal_boot_n_effective,)


@app.cell
def _(bimodal_boot_delta, bimodal_boot_ratio, build_mixture):
    bimodal_boot_scenario = build_mixture(bimodal_boot_delta.value, bimodal_boot_ratio.value)
    return (bimodal_boot_scenario,)


@app.cell
def _(bimodal_boot_scenario, true_dist_ranges):
    bimodal_boot_x_range, bimodal_boot_y_range = true_dist_ranges(
        bimodal_boot_scenario["dist"], p_lo=0.0005, p_hi=0.9995
    )
    return bimodal_boot_x_range, bimodal_boot_y_range


@app.cell
def _(mo):
    bimodal_seed = mo.ui.number(
        start=0, stop=999_999, step=1, value=PAPER.MC_BASE_SEED,
        label="Reference-sample seed",
    )
    # ONE control decides WHICH realization is bootstrapped: the seed box.
    # A "↻ draw a new realization" button used to sit beside it doing the same
    # job by adding an invisible offset to that seed, so two controls moved the
    # same thing and the box stopped showing the seed actually in use. The dice
    # button stays: it does something different -- it holds the realization
    # fixed and draws another resample from it, which is the bootstrap step.
    bimodal_boot_redraw = mo.ui.button(
        label="🎲 Draw another resample", value=0, on_click=lambda v: v + 1)
    return bimodal_boot_redraw, bimodal_seed


@app.cell
def _(bimodal_boot_redraw, bimodal_seed, mo):
    mo.vstack([
        mo.hstack([bimodal_seed, bimodal_boot_redraw], justify="start", gap=2),
        mo.md(
            f"*Reference realization: seed **{int(bimodal_seed.value)}** &nbsp;&middot;&nbsp; "
            f"resample **#{bimodal_boot_redraw.value + 1}***  \n"
            "*The **seed** chooses **which** single sample is being bootstrapped; **🎲** holds that "
            "sample fixed and draws another resample from it.*"
        ),
    ], gap=1)
    return


@app.cell
def _(bimodal_boot_n_effective, bimodal_boot_scenario, bimodal_seed, draw_mixture):
    # draw_mixture takes the SEED, not an rng, because the paper's sampler
    # seeds each mixture component separately.
    _seed = int(bimodal_seed.value)
    bimodal_reference_sample = draw_mixture(_seed, bimodal_boot_scenario, bimodal_boot_n_effective)
    return (bimodal_reference_sample,)


@app.cell
def _(bimodal_boot_n_effective, bimodal_boot_redraw, bimodal_reference_sample, np):
    # REDRAW_BASE, not a bare 70_000: that literal sat inside the fish
    # replicate pool (fish_seed(0.7, b) spans 70_042..71_041).
    _rng = np.random.default_rng(PAPER.REDRAW_BASE + bimodal_boot_redraw.value)
    bimodal_boot_x_sample = np.sort(_rng.choice(bimodal_reference_sample, size=bimodal_boot_n_effective, replace=True))
    _n = len(bimodal_boot_x_sample)
    bimodal_boot_y_sample = np.arange(1, _n + 1) / (_n + 1)
    return bimodal_boot_x_sample, bimodal_boot_y_sample


@app.cell
def _(bimodal_boot_x_sample, hartigan_line_md, mo):
    hartigan_line_md(mo, bimodal_boot_x_sample)
    return


@app.cell
def _(bimodal_boot_constraint, bimodal_boot_k_metalog, bimodal_boot_k_qflex, mo, render_model_controls):
    render_model_controls(mo, bimodal_boot_k_metalog, bimodal_boot_k_qflex, bimodal_boot_constraint)
    return


@app.cell
def _(
    bimodal_boot_constraint,
    bimodal_boot_k_metalog,
    bimodal_boot_k_qflex,
    bimodal_boot_x_sample,
    bimodal_boot_y_sample,
    fit_metalog_qflex,
):
    _res = fit_metalog_qflex(
        bimodal_boot_x_sample, bimodal_boot_y_sample, bimodal_boot_k_metalog.value,
        bimodal_boot_k_qflex.value, bimodal_boot_constraint.value,
    )
    bimodal_boot_fit_error = _res["fit_error"]
    bimodal_boot_metalog_fit, bimodal_boot_metalog_curve, bimodal_boot_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    bimodal_boot_qflex_fit, bimodal_boot_qflex_curve, bimodal_boot_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return bimodal_boot_fit_error, bimodal_boot_metalog_curve, bimodal_boot_metalog_fit, bimodal_boot_metalog_modes, bimodal_boot_qflex_curve, bimodal_boot_qflex_fit, bimodal_boot_qflex_modes


@app.cell
def _(
    PLOTLY_CONFIG,
    bimodal_boot_constraint,
    bimodal_boot_fit_error,
    bimodal_boot_k_metalog,
    bimodal_boot_k_qflex,
    bimodal_boot_metalog_curve,
    bimodal_boot_metalog_fit,
    bimodal_boot_metalog_modes,
    bimodal_boot_qflex_curve,
    bimodal_boot_qflex_fit,
    bimodal_boot_qflex_modes,
    bimodal_boot_scenario,
    bimodal_boot_x_range,
    bimodal_boot_x_sample,
    bimodal_boot_y_range,
    bimodal_boot_y_sample,
    mo,
    render_mc_panel,
):
    render_mc_panel(
        mo, PLOTLY_CONFIG, bimodal_boot_scenario["dist"], bimodal_boot_x_sample, bimodal_boot_y_sample,
        bimodal_boot_k_metalog.value, bimodal_boot_k_qflex.value, bimodal_boot_constraint.value,
        bimodal_boot_metalog_curve, bimodal_boot_metalog_fit, bimodal_boot_metalog_modes,
        bimodal_boot_qflex_curve, bimodal_boot_qflex_fit, bimodal_boot_qflex_modes,
        bimodal_boot_fit_error, bimodal_boot_x_range, bimodal_boot_y_range,
        true_n_modes=bimodal_boot_scenario["true_modes"],
    )
    return


@app.cell
def _(mo):
    bimodal_boot_n_replicates = mo.ui.slider(start=5, stop=1000, step=5, value=PAPER.N_BOOT, label="Replicates", show_value=True)
    bimodal_boot_run_batch = mo.ui.run_button(label="▶ Run Bootstrap Analysis")
    return bimodal_boot_n_replicates, bimodal_boot_run_batch


@app.cell
def _(bimodal_boot_n_replicates, bimodal_boot_run_batch, mo):
    mo.vstack([
        mo.md(
            "**Full bootstrap simulation** — uses this experiment's own mixture, N, and K's; "
            "each replicate resamples the fixed reference realization. W1 is measured against that "
            "**reference sample's own EQF**."
        ),
        mo.hstack([bimodal_boot_n_replicates, bimodal_boot_run_batch], justify="start", gap=2),
    ], gap=1)
    return


@app.cell
def _(
    FIT_P_GRID,
    bimodal_boot_k_metalog,
    bimodal_boot_k_qflex,
    bimodal_boot_n_effective,
    bimodal_boot_n_replicates,
    bimodal_boot_run_batch,
    bimodal_boot_scenario,
    bimodal_reference_sample,
    bimodal_seed,
    mo,
    np,
    run_replicate_batch,
):
    if bimodal_boot_run_batch.value:
        _n_ref = len(bimodal_reference_sample)
        _p_ref = np.arange(1, _n_ref + 1) / (_n_ref + 1)
        _x_ref_grid = np.interp(FIT_P_GRID, _p_ref, bimodal_reference_sample)

        def _draw(rng, rep):
            # Per-replicate stream, so replicate b is reachable on its own
            # instead of only by replaying the loop. This section has no
            # counterpart in the paper, so the base is the section's own.
            _r = np.random.default_rng((bimodal_seed.value + 777) * 10_000 + rep)
            _x = np.sort(_r.choice(bimodal_reference_sample,
                                   size=bimodal_boot_n_effective, replace=True))
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, bimodal_boot_n_replicates.value, bimodal_boot_k_metalog.value, bimodal_boot_k_qflex.value,
            _draw, bimodal_seed.value + 777, w1_ref=_x_ref_grid,
            w1_label="W1 vs reference sample", true_n_modes=bimodal_boot_scenario["true_modes"],
            table_format="bimodal"                 # Table 6 columns; the bootstrap twin has no table of its own
        )
    else:
        mo.output.replace(
            mo.md("*Click **▶ Run Bootstrap Analysis** to resample the fixed reference repeatedly and summarize all 4 QPDs.*")
        )
    return





@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("Empirical case studies", level=2)}

        Four real datasets, each in its own section with its **own**
        Metalog K / QFlex K / constraint controls, its own Hartigan dip
        test, and its own bootstrap batch analysis. Every panel shows the
        empirical quantile function (raw sorted data as a step-like curve)
        with a pointwise 95% bootstrap CI, plus a Metalog/QFlex fit, plus
        (on the density panel) a histogram of the raw data for reference.
        **Drag a rectangle on a plot to zoom**; double-click to reset.

        We begin with a historical asset-class returns dataset as an
        additional real-world example, then the paper's own three
        empirical case studies: river gauge height, fish weights, and the
        Old Faithful geyser.
        """
    )
    return


@app.cell
def _(DATA_DIR, io, np, pd):
    def load_fish_raw():
        _df = pd.read_excel(str(DATA_DIR / "Fish Biology.xlsx"))
        return np.sort(_df["Fish Weight (lbs)"].dropna().values.astype(float))

    def load_hydrology_raw():
        _df = pd.read_excel(str(DATA_DIR / "Hydrology.xlsx"))
        return np.sort(_df["Gauge Height (ft)"].dropna().values.astype(float))

    def load_geyser_raw():
        # Whitespace-separated text with a few "M" (missing) markers in the
        # eruption-duration column (unused here). `str(DATA_DIR / ...)` is
        # an actual local path when running desktop marimo, but resolves to
        # an https:// URL in the deployed WASM build.
        #
        # Handing that URL *string* straight to pd.read_csv() is what broke
        # this loader (and only this loader) in the deployed build: GitHub
        # Pages/Fastly serves plain-text files like this one with a
        # transparent `Content-Encoding: gzip` (the browser already
        # decompresses it before Python ever sees the bytes -- normal,
        # invisible transport-level compression). But when pandas itself
        # fetches a URL (pandas.io.common._get_filepath_or_buffer), it
        # unconditionally re-checks that same now-irrelevant response
        # header and, seeing "gzip", forces ANOTHER round of gzip
        # decompression on the already-plain bytes -- even when
        # compression=None is passed explicitly, since this override
        # happens before the caller's compression argument is consulted.
        # That crashes with `gzip.BadGzipFile: Not a gzipped file`, which
        # aborts this cell and (being upstream of the plot) blanks the
        # entire Geyser section with no plot area and no visible error.
        # The .xlsx loaders above don't hit this because GitHub Pages
        # doesn't gzip already-compressed binary content, only plain text.
        #
        # Fetching the bytes ourselves and handing pandas a BytesIO buffer
        # instead of the URL string sidesteps the bug entirely: pandas'
        # Content-Encoding override only triggers when pandas does the
        # URL fetching itself, not when it's just parsing a buffer.
        # The 299-point Azzalini & Bowman (1990) series. The older
        # geyser.txt held 298 rows: it was missing the first observation and
        # recorded one waiting time as 55 where the source gives 77.
        _path = str(DATA_DIR / "geyser_299.csv")
        if _path.startswith("http://") or _path.startswith("https://"):
            import urllib.request
            with urllib.request.urlopen(_path) as _resp:
                _source = io.BytesIO(_resp.read())
        else:
            _source = _path
        _df = pd.read_csv(_source)
        return np.sort(_df["waiting"].dropna().values.astype(float))

    def load_returns_df():
        _df = pd.read_excel(str(DATA_DIR / "Returns-Stocks_Bonds_Bills.xlsx"))
        _df.columns = [c.strip() for c in _df.columns]
        return _df

    def eqf_bootstrap_ci(x_raw, p_grid, n_boot, seed=42, boot_source=None, jitter=0.0):
        # Pointwise 95% percentile bootstrap CI on the EQF, matching the
        # method used for the paper's own EQF+CI figures (the river-gauge
        # EQF with bootstrap intervals): resample the raw data with replacement, interpolate
        # each resample's EQF onto a common probability grid, and take the
        # 2.5th/97.5th percentiles at each grid point across resamples.
        #
        # `boot_source` / `jitter` exist for the fish-weight case,
        # where the recorded values are heavily rounded and the quantity of
        # interest is the *latent* (pre-rounding) weight. The paper handles
        # this by adding "±0.5 lb uniform jitter to the bootstrap
        # resamples" -- i.e. resample the raw, rounded values and add a
        # FRESH jitter draw to each resample, so the de-rounding
        # uncertainty is part of what the CI measures. Jittering once up
        # front and then resampling that single jittered array (the obvious
        # shortcut) is a different and wrong procedure: it freezes one
        # arbitrary tie-breaking into every replicate and reports the noise
        # as if it were data.
        #
        # JITTER CONVENTION: `jitter` is the HALF-WIDTH (amplitude), so the
        # noise is Uniform(-jitter, +jitter) and jitter=0.5 is the paper's
        # "±0.5 lb". This matches every script in the repro repo
        # (bootstrap_curve_store.py, fish_median_w1_bootstrap_ci.py,
        # fish_mode_scatter_k10.py, ...), so a number set here means the same
        # thing as the same number there. An earlier version of this notebook
        # took the argument as the FULL width and halved it internally, which
        # made "0.5" here only half the noise of "0.5" in the repo.
        _N = len(x_raw)
        _p_emp = np.arange(1, _N + 1) / (_N + 1)
        _src = x_raw if boot_source is None else np.asarray(boot_source, dtype=float)
        _rng = np.random.default_rng(seed)
        _boot = np.empty((n_boot, len(p_grid)))
        for _b in range(n_boot):
            _draw = _rng.choice(_src, size=_N, replace=True)
            if jitter > 0:
                _draw = _draw + _rng.uniform(-jitter, jitter, size=_N)
            _x_boot = np.sort(_draw)
            _boot[_b] = np.interp(p_grid, _p_emp, _x_boot)
        _point = np.interp(p_grid, _p_emp, x_raw)
        _lo = np.percentile(_boot, 2.5, axis=0)
        _hi = np.percentile(_boot, 97.5, axis=0)
        return _point, _lo, _hi

    return eqf_bootstrap_ci, load_fish_raw, load_geyser_raw, load_hydrology_raw, load_returns_df


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("Historical asset-class returns", level=3)}

        A public dataset of historical annual returns across 7 U.S. asset
        classes &mdash; S&amp;P 500 (with dividends), small-cap stocks,
        3-month T-bills, 10-year T-bonds, Baa corporate bonds, real
        estate, and gold &mdash; 1928-2025 (N=98 years each), matching the
        widely used historical-returns series published by NYU Stern's
        Aswath Damodaran. This dataset isn't one of the paper's own three
        empirical case studies below; it's included here as an additional
        real-world example. Pick a category below, then work through the
        same Metalog/QFlex fit, EQF+CI plot, Hartigan dip test, and
        bootstrap batch analysis as the sections that follow.

        Annual returns can be negative, so this section fits plain
        (unbounded) Metalog / QFlex throughout, unlike the semi-bounded
        fits used for the three nonnegative case studies below.
        """
    )
    return


@app.cell
def _(load_returns_df):
    returns_df = load_returns_df()
    returns_categories = [c for c in returns_df.columns if c.lower() != "year"]
    return returns_categories, returns_df


@app.cell
def _(mo, returns_categories):
    returns_category = mo.ui.dropdown(
        options=returns_categories, value=returns_categories[0], label="Return category",
    )
    returns_category
    return (returns_category,)


@app.cell
def _(returns_category, returns_df, np):
    returns_x = np.sort(returns_df[returns_category.value].dropna().to_numpy(dtype=float))
    _n = len(returns_x)
    returns_y = np.arange(1, _n + 1) / (_n + 1)  # Weibull plotting position, matching the paper's Equation 3
    return returns_x, returns_y


@app.cell
def _(hartigan_line_md, mo, returns_x):
    hartigan_line_md(mo, returns_x)
    return


@app.cell
def _(mo):
    returns_k_metalog = mo.ui.slider(start=2, stop=15, step=1, value=7, label="Metalog K", show_value=True)
    returns_k_qflex = mo.ui.slider(start=2, stop=15, step=1, value=7, label="QFlex K", show_value=True)
    returns_qflex_constraint = mo.ui.dropdown(
        options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
        value="Unconstrained", label="QFlex constraint",
    )
    mo.vstack([mo.hstack([returns_k_metalog, returns_k_qflex], justify="start", gap=2), returns_qflex_constraint])
    return returns_k_metalog, returns_k_qflex, returns_qflex_constraint


@app.cell
def _(eqf_bootstrap_ci, returns_x, np):
    returns_p_grid = np.linspace(0.01, 0.99, len(PAPER.P_GRID))
    returns_eqf_point, returns_eqf_lo, returns_eqf_hi = eqf_bootstrap_ci(returns_x, returns_p_grid, n_boot=PAPER.N_BOOT, seed=42)
    return returns_eqf_hi, returns_eqf_lo, returns_eqf_point, returns_p_grid


@app.cell
def _(make_normal_reference, returns_x):
    # Normal baseline: the default model most return analyses start from,
    # and a two-parameter one that cannot produce extra modes -- so any
    # multimodality a QPD reports here is visibly the QPD's doing, not the
    # data's. Refits whenever the asset-class dropdown changes.
    returns_normal = make_normal_reference(returns_x)
    return (returns_normal,)


@app.cell
def _(mo, returns_category, returns_normal, returns_x, np):
    _p = returns_normal["params"]
    _skew = float(np.mean(((returns_x - np.mean(returns_x)) / np.std(returns_x)) ** 3))
    _exkurt = float(np.mean(((returns_x - np.mean(returns_x)) / np.std(returns_x)) ** 4)) - 3.0
    mo.md(
        f"**Normal reference fit** for {returns_category.value}: μ = {_p['mu']:.4f}, σ = {_p['sigma']:.4f} "
        f"(sample skewness {_skew:+.2f}, excess kurtosis {_exkurt:+.2f}). Drawn as a dashed curve "
        "alongside the QPD fits below — being a two-parameter symmetric model it is unimodal by "
        "construction, so it is the baseline against which any extra structure a QPD reports "
        "should be read."
    )
    return


@app.cell
def _(fit_metalog_qflex, returns_k_metalog, returns_k_qflex, returns_qflex_constraint, returns_x, returns_y):
    _res = fit_metalog_qflex(
        returns_x, returns_y, returns_k_metalog.value, returns_k_qflex.value, returns_qflex_constraint.value
    )
    returns_fit_error = _res["fit_error"]
    returns_metalog_fit, returns_metalog_curve, returns_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    returns_qflex_fit, returns_qflex_curve, returns_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return returns_fit_error, returns_metalog_curve, returns_metalog_fit, returns_metalog_modes, returns_qflex_curve, returns_qflex_fit, returns_qflex_modes


@app.cell
def _(
    PLOTLY_CONFIG,
    returns_category,
    returns_eqf_hi,
    returns_eqf_lo,
    returns_eqf_point,
    returns_fit_error,
    returns_metalog_curve,
    returns_metalog_fit,
    returns_metalog_modes,
    returns_normal,
    returns_p_grid,
    returns_qflex_constraint,
    returns_qflex_curve,
    returns_qflex_fit,
    returns_qflex_modes,
    returns_x,
    mo,
    render_empirical_panel,
):
    render_empirical_panel(
        mo, PLOTLY_CONFIG, returns_category.value, "Annual return", returns_qflex_constraint.value, returns_p_grid,
        returns_eqf_point, returns_eqf_lo, returns_eqf_hi, returns_x, returns_metalog_curve, returns_metalog_fit,
        returns_metalog_modes, returns_qflex_curve, returns_qflex_fit, returns_qflex_modes, returns_fit_error,
        reference_fits=[returns_normal],
    )
    return


@app.cell
def _(mo):
    returns_n_replicates = mo.ui.slider(start=5, stop=1000, step=5, value=PAPER.N_BOOT, label="Replicates", show_value=True)
    returns_run_batch = mo.ui.run_button(label="▶ Run Bootstrap Analysis")
    mo.md("**Full simulation for this category** — bootstrap-resample the annual returns and refit repeatedly, across all 4 QPDs.")
    mo.hstack([returns_n_replicates, returns_run_batch], justify="start", gap=2)
    return returns_n_replicates, returns_run_batch


@app.cell
def _(
    fit_all_qpds,
    returns_k_metalog,
    returns_k_qflex,
    returns_n_replicates,
    returns_run_batch,
    returns_x,
    mo,
    np,
    run_replicate_batch,
):
    if returns_run_batch.value:
        _all_fits, _ = fit_all_qpds(returns_x, np.arange(1, len(returns_x) + 1) / (len(returns_x) + 1), returns_k_metalog.value, returns_k_qflex.value)
        _w1_refs = {_label: (_r["curve"][0] if _r["curve"] is not None else None) for _label, _r in _all_fits.items()}

        def _draw(rng, rep):
            # Per-replicate stream (see the bimodal bootstrap above). The
            # asset-return section is the notebook's own extension, not part of
            # the paper, so the base seed is local to it.
            _r = np.random.default_rng(PAPER.RETURNS_BASE + rep)
            _x = np.sort(_r.choice(returns_x, size=len(returns_x), replace=True))
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, returns_n_replicates.value, returns_k_metalog.value, returns_k_qflex.value,
            _draw, PAPER.RETURNS_BASE, w1_ref=_w1_refs, w1_label="W1 vs full-sample fit",
            table_format="bootstrap"               # notebook-only section: no paper table
        )
    else:
        mo.output.replace(mo.md("*Click **▶ Run Bootstrap Analysis** to bootstrap-resample and refit repeatedly.*"))
    return


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("River gauge height", level=3)}

        Annual peak flood gauge heights for the Williamson River, N=95
        years (1920&ndash;2014), as reported by the U.S. Geological Survey
        (USGS) and used by Keelin (2016) as an illustrative QPD example.
        As annual block maxima, this dataset sits squarely in the
        traditional domain of extreme-value theory &mdash; the paper's
        case where a stable model fit and the underlying theory both
        point toward unimodal structure.

        **Boundedness:** gauge height can't be negative, so &mdash;
        matching the paper's own convention &mdash; this section fits
        **Log Metalog / Log QFlex** (semi-bounded, lower bound = 0)
        rather than the unbounded variants.
        """
    )
    return


@app.cell
def _(mo, paper_settings_note):
    paper_settings_note(
        mo,
        "`42 + 40,000,000 + b`. GEV and every QPD see the **same** replicate *b*, so the curves are paired; the draft's own scripts drew GEV from seed 42 and the QPDs from 43.",
        [("Metalog K", "10"), ("QFlex K", "10"), ("QFlex constraint", "A+"), ("Jitter", "none &mdash; gauge heights are continuous"), ("Replicates", "1000")],
        "Table 7 and the river-gauge figures. At these settings Log Metalog K=10 gives 37.2 % valid and median W1 0.0453; Log QFlex-A+ K=10 gives 100 % and 0.0467.",
    )
    return

@app.cell
def _(load_hydrology_raw, np):
    hydro_x = load_hydrology_raw()
    _n = len(hydro_x)
    hydro_y = np.arange(1, _n + 1) / (_n + 1)
    return hydro_x, hydro_y


@app.cell
def _(hartigan_line_md, hydro_x, mo):
    hartigan_line_md(mo, hydro_x)
    return


@app.cell
def _(mo):
    hydro_k_metalog = mo.ui.slider(start=2, stop=15, step=1, value=10, label="Metalog K", show_value=True)
    hydro_k_qflex = mo.ui.slider(start=2, stop=15, step=1, value=10, label="QFlex K", show_value=True)
    # Hydrology: the paper compares GEV and Log Metalog K=10 against
    # Log QFlex-A+ K=10 (A+ enforces unimodality).
    hydro_qflex_constraint = mo.ui.dropdown(
        options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
        value="A+  (all coefficients ≥ 0)", label="QFlex constraint",
    )
    mo.vstack([mo.hstack([hydro_k_metalog, hydro_k_qflex], justify="start", gap=2), hydro_qflex_constraint])
    return hydro_k_metalog, hydro_k_qflex, hydro_qflex_constraint


@app.cell
def _(eqf_bootstrap_ci, hydro_x, np):
    hydro_p_grid = np.linspace(0.01, 0.99, len(PAPER.P_GRID))
    hydro_eqf_point, hydro_eqf_lo, hydro_eqf_hi = eqf_bootstrap_ci(hydro_x, hydro_p_grid, n_boot=PAPER.N_BOOT, seed=42)
    return hydro_eqf_hi, hydro_eqf_lo, hydro_eqf_point, hydro_p_grid


@app.cell
def _(hydro_x, make_gev_reference):
    # GEV baseline: these are annual block maxima, the textbook case for
    # extreme-value theory, and the paper fits a GEV here precisely to have
    # a theory-backed, stable reference for the mode against which the
    # high-order Log Metalog's unstable spike can be judged.
    hydro_gev = make_gev_reference(hydro_x)
    return (hydro_gev,)


@app.cell
def _(hydro_gev, mo):
    _p = hydro_gev["params"]
    _tail = ("Fréchet — heavy upper tail" if _p["xi"] > 0.02
             else ("Weibull — bounded upper tail" if _p["xi"] < -0.02 else "Gumbel — exponential tail"))
    mo.md(
        f"**GEV reference fit:** ξ = {_p['xi']:.3f}, μ = {_p['loc']:.3f}, σ = {_p['scale']:.3f} "
        f"({_tail}). Fitted by maximum likelihood to the 95 annual maxima and drawn as a dashed "
        "curve alongside the QPD fits below; being a three-parameter extreme-value model it is "
        "unimodal by construction, so it cannot manufacture the extra modes a high-order QPD can."
    )
    return


@app.cell
def _(fit_metalog_qflex, hydro_k_metalog, hydro_k_qflex, hydro_qflex_constraint, hydro_x, hydro_y):
    _res = fit_metalog_qflex(hydro_x, hydro_y, hydro_k_metalog.value, hydro_k_qflex.value, hydro_qflex_constraint.value, bounds=(0, None))
    hydro_fit_error = _res["fit_error"]
    hydro_metalog_fit, hydro_metalog_curve, hydro_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    hydro_qflex_fit, hydro_qflex_curve, hydro_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return hydro_fit_error, hydro_metalog_curve, hydro_metalog_fit, hydro_metalog_modes, hydro_qflex_curve, hydro_qflex_fit, hydro_qflex_modes


@app.cell
def _(
    PLOTLY_CONFIG,
    hydro_eqf_hi,
    hydro_eqf_lo,
    hydro_eqf_point,
    hydro_fit_error,
    hydro_gev,
    hydro_metalog_curve,
    hydro_metalog_fit,
    hydro_metalog_modes,
    hydro_p_grid,
    hydro_qflex_constraint,
    hydro_qflex_curve,
    hydro_qflex_fit,
    hydro_qflex_modes,
    hydro_x,
    mo,
    render_empirical_panel,
):
    render_empirical_panel(
        mo, PLOTLY_CONFIG, "River gauge height", "Gauge height (ft)", hydro_qflex_constraint.value, hydro_p_grid,
        hydro_eqf_point, hydro_eqf_lo, hydro_eqf_hi, hydro_x, hydro_metalog_curve, hydro_metalog_fit,
        hydro_metalog_modes, hydro_qflex_curve, hydro_qflex_fit, hydro_qflex_modes, hydro_fit_error,
        bounds=(0, None), reference_fits=[hydro_gev],
    )
    return


@app.cell
def _(mo):
    hydro_n_replicates = mo.ui.slider(start=5, stop=1000, step=5, value=PAPER.N_BOOT, label="Replicates", show_value=True)
    hydro_run_batch = mo.ui.run_button(label="▶ Run Bootstrap Analysis")
    mo.md("**Full simulation for this dataset** — bootstrap-resample the gauge-height data and refit repeatedly, across all 4 QPDs.")
    mo.hstack([hydro_n_replicates, hydro_run_batch], justify="start", gap=2)
    return hydro_n_replicates, hydro_run_batch


@app.cell
def _(
    fit_all_qpds,
    hydro_k_metalog,
    hydro_k_qflex,
    hydro_n_replicates,
    hydro_run_batch,
    hydro_x,
    hydro_y,
    hydro_p_grid,
    hydro_eqf_point,
    FIT_P_GRID,
    PAPER,
    mo,
    np,
    run_replicate_batch,
):
    if hydro_run_batch.value:
        _all_fits, _ = fit_all_qpds(hydro_x, hydro_y, hydro_k_metalog.value, hydro_k_qflex.value, bounds=(0, None))
        _w1_refs = {_label: (_r["curve"][0] if _r["curve"] is not None else None) for _label, _r in _all_fits.items()}

        def _draw(rng, rep):
            # The paper's resampling: per-replicate seed, so replicate b is
            # the same sample here as in the reproduction scripts. Gauge
            # heights are continuous, so no jitter.
            _x = np.sort(PAPER.hydro_resample(hydro_x, b=rep))
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, hydro_n_replicates.value, hydro_k_metalog.value, hydro_k_qflex.value,
            _draw, PAPER.hydro_seed(b=0),
            # Equation 6 measures against the OBSERVED sample's EQF, not
            # against the full-sample fit.
            w1_ref=np.interp(FIT_P_GRID, hydro_p_grid, hydro_eqf_point),
            w1_label="W1 vs empirical", bounds=(0, None),
            table_format="hydrology"               # Table 7
        )
    else:
        mo.output.replace(mo.md("*Click **▶ Run Bootstrap Analysis** to bootstrap-resample and refit repeatedly.*"))
    return


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("Fish weights", level=3)}

        Weight measurements of N=3,474 steelhead trout from the Babine
        River in northern British Columbia, used by Keelin (2016) to
        illustrate how high-order Metalogs can uncover multimodality in
        empirical data. The raw weights are heavily rounded (about 91%
        recorded as whole pounds), making this the paper's case where true
        modality remains genuinely data-dependent and unresolved.

        **Boundedness:** weight can't be negative, so &mdash; matching the
        paper's own convention &mdash; this section fits **Log Metalog /
        Log QFlex** (semi-bounded, lower bound = 0) rather than the
        unbounded variants.
        """
    )
    return


@app.cell
def _(mo, paper_settings_note):
    paper_settings_note(
        mo,
        "`42 + round(j×100)×10,000 + b`; at the default jitter j = 0.5 that is `500042 + b`. Jitter is the **half-width**, so j means Uniform(−j, +j), and each replicate is re-jittered independently.",
        [("Metalog K", "10"), ("QFlex K", "10"), ("QFlex constraint", "TA+"), ("Jitter half-width j", "0.5 lb"), ("Replicates", "1000")],
        "Tables 8 and 9, and the fish-weight figures. At these settings 82.0 % of Log Metalog K=10 fits are bimodal against 28.2 % for Log QFlex-TA+.",
    )
    return

@app.cell
def _(mo):
    fish_jitter = mo.ui.slider(
        start=0.0, stop=1.0, step=0.05, value=0.5,
        label="Jitter half-width j (± lbs)", show_value=True,
    )
    mo.vstack([
        fish_jitter,
        mo.md(
            "*Fish weights are heavily rounded — 90.8% are whole pounds and "
            "the rest fall on half-pounds, giving only 56 distinct values "
            "across 3,474 observations. Treating the quantity of interest as "
            "the **latent** (pre-rounding) weight, this slider adds "
            "`Uniform(-j, +j)` noise to undo that rounding; j = 0 uses the "
            "raw, recorded data. The slider is the **half-width**, so j = 0.5 "
            "spans a full pound — the paper's «±0.5 lb», and the same meaning "
            "`jitter=0.5` has in the repro scripts.*  \n"
            "*The fit shown below uses one fixed jitter realization, but the "
            "bootstrap CI and the batch analysis resample the **raw** weights "
            "and re-jitter every replicate independently — matching the "
            "paper's «±0.5 lb uniform jitter to the bootstrap resamples», so "
            "the de-rounding uncertainty is part of what they measure.*"
        ),
    ])
    return (fish_jitter,)


@app.cell
def _(PAPER, fish_jitter, load_fish_raw, np):
    # `fish_raw` keeps the recorded (heavily rounded) weights; `fish_x` is
    # the single jittered realization that the displayed point-estimate fit
    # is computed from. The bootstrap below deliberately does NOT resample
    # `fish_x` -- it resamples `fish_raw` and re-jitters each replicate, per
    # the paper's "±0.5 lb uniform jitter to the bootstrap resamples".
    fish_raw = load_fish_raw()
    if fish_jitter.value > 0:
        # Draw 0 of the paper's own jitter_original stream -- the same
        # realization the paper's fish figures are drawn from. This used to be
        # default_rng(20260828), a date with no relation to any other seed in
        # the project, which put the displayed point-estimate fit on a jitter
        # realization that appears nowhere in the paper. (20_260_828 also lands
        # inside the geyser replicate pool at a 2.6-minute jitter.)
        fish_x = np.sort(PAPER.jitter_original(
            fish_raw, fish_jitter.value, PAPER.FISH_STREAM_BASE,
            draw=0, floor=PAPER.FISH_FLOOR))
    else:
        fish_x = fish_raw
    _n = len(fish_x)
    fish_y = np.arange(1, _n + 1) / (_n + 1)
    return fish_raw, fish_x, fish_y


@app.cell
def _(fish_x, hartigan_line_md, mo):
    hartigan_line_md(mo, fish_x)
    return


@app.cell
def _(mo):
    fish_k_metalog = mo.ui.slider(start=2, stop=15, step=1, value=10, label="Metalog K", show_value=True)
    fish_k_qflex = mo.ui.slider(start=2, stop=15, step=1, value=10, label="QFlex K", show_value=True)
    # Fish: the paper uses Log QFlex-TA+ (not A+) because the data give
    # real evidence against unimodality, so a QPD able to represent two
    # modes is appropriate. Headline order K=10.
    fish_qflex_constraint = mo.ui.dropdown(
        options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
        value="TA+  (tail coefficients ≥ 0)", label="QFlex constraint",
    )
    mo.vstack([mo.hstack([fish_k_metalog, fish_k_qflex], justify="start", gap=2), fish_qflex_constraint])
    return fish_k_metalog, fish_k_qflex, fish_qflex_constraint


@app.cell
def _(eqf_bootstrap_ci, fish_jitter, fish_raw, fish_x, np):
    fish_p_grid = np.linspace(0.01, 0.99, len(PAPER.P_GRID))
    # Point estimate from the displayed jittered sample; CI from resampling
    # the RAW rounded weights with a fresh jitter draw per replicate, so the
    # band reflects de-rounding uncertainty rather than one frozen tie-break.
    fish_eqf_point, fish_eqf_lo, fish_eqf_hi = eqf_bootstrap_ci(
        fish_x, fish_p_grid, n_boot=PAPER.N_BOOT, seed=42,
        boot_source=fish_raw, jitter=fish_jitter.value,
    )
    return fish_eqf_hi, fish_eqf_lo, fish_eqf_point, fish_p_grid


@app.cell
def _(fish_k_metalog, fish_k_qflex, fish_qflex_constraint, fish_x, fish_y, fit_metalog_qflex):
    _res = fit_metalog_qflex(fish_x, fish_y, fish_k_metalog.value, fish_k_qflex.value, fish_qflex_constraint.value, bounds=(0, None))
    fish_fit_error = _res["fit_error"]
    fish_metalog_fit, fish_metalog_curve, fish_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    fish_qflex_fit, fish_qflex_curve, fish_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return fish_fit_error, fish_metalog_curve, fish_metalog_fit, fish_metalog_modes, fish_qflex_curve, fish_qflex_fit, fish_qflex_modes


@app.cell
def _(
    PAPER,
    PLOTLY_CONFIG,
    fish_eqf_hi,
    fish_eqf_lo,
    fish_eqf_point,
    fish_fit_error,
    fish_metalog_curve,
    fish_metalog_fit,
    fish_metalog_modes,
    fish_p_grid,
    fish_qflex_constraint,
    fish_qflex_curve,
    fish_qflex_fit,
    fish_qflex_modes,
    fish_x,
    mo,
    render_empirical_panel,
):
    render_empirical_panel(
        mo, PLOTLY_CONFIG, "Fish weights", "Weight (lbs)", fish_qflex_constraint.value, fish_p_grid, fish_eqf_point,
        fish_eqf_lo, fish_eqf_hi, fish_x, fish_metalog_curve, fish_metalog_fit, fish_metalog_modes, fish_qflex_curve,
        fish_qflex_fit, fish_qflex_modes, fish_fit_error, value_xlim=(None, 28), bounds=(0, None),
        bin_width=2 * PAPER.FISH_JITTER, bin_start=0.5,
        true_n_modes=None,
    )
    return


@app.cell
def _(mo):
    fish_n_replicates = mo.ui.slider(start=5, stop=1000, step=5, value=PAPER.N_BOOT, label="Replicates", show_value=True)
    fish_run_batch = mo.ui.run_button(label="▶ Run Bootstrap Analysis")
    mo.md("**Full simulation for this dataset** — bootstrap-resample the raw fish weights, re-jitter each replicate, and refit repeatedly, across all 4 QPDs.")
    mo.hstack([fish_n_replicates, fish_run_batch], justify="start", gap=2)
    return fish_n_replicates, fish_run_batch


@app.cell
def _(
    fish_jitter,
    fish_k_metalog,
    fish_k_qflex,
    fish_n_replicates,
    fish_raw,
    fish_run_batch,
    fish_x,
    fish_y,
    fish_p_grid,
    fish_eqf_point,
    FIT_P_GRID,
    PAPER,
    fit_all_qpds,
    mo,
    np,
    run_replicate_batch,
):
    if fish_run_batch.value:
        _all_fits, _ = fit_all_qpds(fish_x, fish_y, fish_k_metalog.value, fish_k_qflex.value, bounds=(0, None))
        _w1_refs = {_label: (_r["curve"][0] if _r["curve"] is not None else None) for _label, _r in _all_fits.items()}

        def _draw(rng, rep):
            # Resample the RAW rounded weights and re-jitter each replicate,
            # per the paper's "±0.5 lb uniform jitter to the bootstrap
            # resamples", using the paper's per-replicate seed so replicate b
            # is the same sample as in the reproduction scripts.
            _x = np.sort(PAPER.fish_resample(fish_raw, fish_jitter.value, rep))
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, fish_n_replicates.value, fish_k_metalog.value, fish_k_qflex.value,
            _draw, PAPER.fish_seed(fish_jitter.value, 0),
            w1_ref=np.interp(FIT_P_GRID, fish_p_grid, fish_eqf_point),
            w1_label="W1 vs empirical", bounds=(0, None),
            true_n_modes=None,
            table_format="fish"                    # Table 9
        )
    else:
        mo.output.replace(mo.md("*Click **▶ Run Bootstrap Analysis** to bootstrap-resample and refit repeatedly.*"))
    return


@app.cell
def _(mo, section_header_html):
    mo.md(
        rf"""
        {section_header_html("The Old Faithful Geyser", level=3)}

        Waiting times between successive eruptions of the Old Faithful
        geyser at Yellowstone &mdash; the ~299-point Azzalini &amp; Bowman
        (1990) sample recorded August 1&ndash;15, 1985. This is one of the
        most famous empirical datasets in statistics, and the paper's case
        where bimodality is directly observable and well established, so
        the open question is the stability of the fitted modes rather than
        their existence.

        **Boundedness:** matching the paper, which "tested the
        semi-bounded QPDs, Log Metalog and Log QFlex-TA+, with lower bound
        set to zero" for this dataset, this section fits **Log Metalog /
        Log QFlex** (semi-bounded, lower bound = 0) rather than the
        unbounded variants.
        """
    )
    return


@app.cell
def _(mo, paper_settings_note):
    paper_settings_note(
        mo,
        "`42 + 20,000,000 + 50×10,000 + b` = `20,500,042 + b`. The recorded waiting times are whole minutes, so each resample gets ±0.5 min of jitter.",
        [("Data", "299 Old Faithful waiting times, Azzalini &amp; Bowman (1990)"), ("Metalog K", "8"), ("QFlex K", "10"), ("QFlex constraint", "TA+"), ("Jitter half-width", "0.5 min"), ("Replicates", "1000")],
        "Table 10 and the geyser figures. At these settings Log Metalog K=8 gives 72.5 % valid with 80.6 % bimodal; Log QFlex-TA+ K=10 gives 99.7 % and 96.8 %.",
    )
    return

@app.cell
def _(load_geyser_raw, np):
    geyser_x = load_geyser_raw()
    _n = len(geyser_x)
    geyser_y = np.arange(1, _n + 1) / (_n + 1)
    return geyser_x, geyser_y


@app.cell
def _(geyser_x, hartigan_line_md, mo):
    hartigan_line_md(mo, geyser_x)
    return


@app.cell
def _(mo):
    geyser_k_metalog = mo.ui.slider(start=2, stop=15, step=1, value=8, label="Metalog K", show_value=True)
    geyser_k_qflex = mo.ui.slider(start=2, stop=15, step=1, value=10, label="QFlex K", show_value=True)
    # Geyser: Log Metalog K=8 and Log QFlex-TA+ K=10 are the pairing in
    # the paper's mode-dispersion table.
    geyser_qflex_constraint = mo.ui.dropdown(
        options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
        value="TA+  (tail coefficients ≥ 0)", label="QFlex constraint",
    )
    mo.vstack([mo.hstack([geyser_k_metalog, geyser_k_qflex], justify="start", gap=2), geyser_qflex_constraint])
    return geyser_k_metalog, geyser_k_qflex, geyser_qflex_constraint


@app.cell
def _(eqf_bootstrap_ci, geyser_x, np):
    geyser_p_grid = np.linspace(0.01, 0.99, len(PAPER.P_GRID))
    # The recorded waiting times are whole minutes (52 distinct values in
    # 299 observations), so the paper adds +/-0.5 min uniform jitter to the
    # bootstrap resamples. Without it the resamples are riddled with ties,
    # which is exactly what drives spurious high-order modality.
    geyser_eqf_point, geyser_eqf_lo, geyser_eqf_hi = eqf_bootstrap_ci(
        geyser_x, geyser_p_grid, n_boot=PAPER.N_BOOT, seed=42,
        jitter=PAPER.GEYSER_JITTER)
    return geyser_eqf_hi, geyser_eqf_lo, geyser_eqf_point, geyser_p_grid


@app.cell
def _(fit_metalog_qflex, geyser_k_metalog, geyser_k_qflex, geyser_qflex_constraint, geyser_x, geyser_y):
    _res = fit_metalog_qflex(
        geyser_x, geyser_y, geyser_k_metalog.value, geyser_k_qflex.value, geyser_qflex_constraint.value,
        bounds=(0, None),
    )
    geyser_fit_error = _res["fit_error"]
    geyser_metalog_fit, geyser_metalog_curve, geyser_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    geyser_qflex_fit, geyser_qflex_curve, geyser_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return geyser_fit_error, geyser_metalog_curve, geyser_metalog_fit, geyser_metalog_modes, geyser_qflex_curve, geyser_qflex_fit, geyser_qflex_modes


@app.cell
def _(
    PAPER,
    PLOTLY_CONFIG,
    geyser_eqf_hi,
    geyser_eqf_lo,
    geyser_eqf_point,
    geyser_fit_error,
    geyser_metalog_curve,
    geyser_metalog_fit,
    geyser_metalog_modes,
    geyser_p_grid,
    geyser_qflex_constraint,
    geyser_qflex_curve,
    geyser_qflex_fit,
    geyser_qflex_modes,
    geyser_x,
    mo,
    render_empirical_panel,
):
    render_empirical_panel(
        mo, PLOTLY_CONFIG, "Old Faithful waiting time", "Waiting time (min)", geyser_qflex_constraint.value,
        geyser_p_grid, geyser_eqf_point, geyser_eqf_lo, geyser_eqf_hi, geyser_x, geyser_metalog_curve,
        geyser_metalog_fit, geyser_metalog_modes, geyser_qflex_curve, geyser_qflex_fit, geyser_qflex_modes,
        geyser_fit_error, value_xlim=(30, None), bounds=(0, None), true_n_modes=2,
        bin_width=2 * PAPER.GEYSER_JITTER, bin_start=0.5,
    )
    return


@app.cell
def _(mo):
    geyser_n_replicates = mo.ui.slider(start=5, stop=1000, step=5, value=PAPER.N_BOOT, label="Replicates", show_value=True)
    geyser_run_batch = mo.ui.run_button(label="▶ Run Bootstrap Analysis")
    mo.md("**Full simulation for this dataset** — bootstrap-resample the waiting times and refit repeatedly, across all 4 QPDs.")
    mo.hstack([geyser_n_replicates, geyser_run_batch], justify="start", gap=2)
    return geyser_n_replicates, geyser_run_batch


@app.cell
def _(
    fit_all_qpds,
    geyser_k_metalog,
    geyser_k_qflex,
    geyser_n_replicates,
    geyser_run_batch,
    geyser_x,
    geyser_y,
    geyser_p_grid,
    geyser_eqf_point,
    FIT_P_GRID,
    PAPER,
    mo,
    np,
    run_replicate_batch,
):
    if geyser_run_batch.value:
        _all_fits, _ = fit_all_qpds(geyser_x, geyser_y, geyser_k_metalog.value, geyser_k_qflex.value, bounds=(0, None))
        _w1_refs = {_label: (_r["curve"][0] if _r["curve"] is not None else None) for _label, _r in _all_fits.items()}

        def _draw(rng, rep):
            # Waiting times are whole minutes, so the paper adds ±0.5 min
            # jitter to each resample. Without it the resamples are full of
            # ties, which is what drives spurious high-order modality.
            _x = np.sort(PAPER.geyser_resample(geyser_x, b=rep))
            return _x, PAPER.weibull(len(_x))

        run_replicate_batch(
            mo, geyser_n_replicates.value, geyser_k_metalog.value, geyser_k_qflex.value,
            _draw, PAPER.geyser_seed(b=0),
            w1_ref=np.interp(FIT_P_GRID, geyser_p_grid, geyser_eqf_point),
            w1_label="W1 vs empirical", bounds=(0, None),
            true_n_modes=2,
            table_format="geyser"                  # Table 10
        )
    else:
        mo.output.replace(mo.md("*Click **▶ Run Bootstrap Analysis** to bootstrap-resample and refit repeatedly.*"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        *Built as a companion to Khanna &amp; Bickel, "Bump Hunting, Structural
        Overfitting, and Quantile-Parameterized Distributions."*
        """
    )
    return


if __name__ == "__main__":
    app.run()
