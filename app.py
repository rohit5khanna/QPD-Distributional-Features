# /// script
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "openpyxl",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="QPD Playground")


@app.cell
def _():
    import sys
    import os

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
    from common.metalog.metalog_v2 import Metalog, MetalogError
    from common.qflex.core import QFlex, ConstraintType
    from common.qflex.constraints import QFlexError
    from common.jpse.johnson import JohnsonSU, JohnsonSL, JohnsonSB
    from common.mode_utils import detect_modes_from_arrays

    DATA_DIR = mo.notebook_location() / "public"
    return (
        ConstraintType,
        DATA_DIR,
        JohnsonSB,
        JohnsonSL,
        JohnsonSU,
        Metalog,
        MetalogError,
        QFlex,
        QFlexError,
        detect_modes_from_arrays,
        go,
        make_subplots,
        mo,
        np,
        os,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        # QPD Playground

        A hands-on companion to *Inferring Distributional Features based on
        Quantile-Parameterized Distribution Fits* (Khanna & Bickel). Every
        panel below is live: drag the sliders, click the buttons, and watch
        the Metalog and QFlex fits move &mdash; including the spurious modes
        that motivate the paper.

        **On this page:** Johnson distributions &middot; Monte Carlo &amp;
        Bootstrap refit &middot; a bimodal-mixture playground &middot; three
        empirical case studies (each with its own QPD fit). Every section
        has its own per-scenario batch-simulation run with the paper's
        feasibility / false-modality / W1 statistics.
        """
    )
    return


@app.cell
def _(ConstraintType, Metalog, MetalogError, QFlex, QFlexError, detect_modes_from_arrays, go, make_subplots, np, pd):
    # Shared helpers used by every fitting/plotting/batch-simulation section
    # below, so the core logic exists in exactly one place instead of being
    # copy-pasted per section.
    FIT_P_GRID = np.linspace(0.01, 0.99, 1000)

    # Paper convention: the constrained variants of QFlex get their own name,
    # not a generic "QFlex" label that hides which constraint was actually
    # used.
    QFLEX_LABELS = {"NONE": "QFlex-U", "A": "QFlex-A+", "TA": "QFlex-TA+"}

    def fit_metalog_qflex(x_sorted, y_plot_pos, k_metalog_val, k_qflex_val, constraint_label):
        """Fit Metalog and QFlex to one (x, y) EQF sample. Never raises."""
        _fit_error = None
        _metalog_fit = _qflex_fit = None
        _metalog_curve = _qflex_curve = None
        _metalog_modes = _qflex_modes = (None, None, None)

        try:
            _mf = Metalog(x_sorted, y_plot_pos, terms=k_metalog_val)
            _xg = _mf.quantile(FIT_P_GRID)
            _pg = _mf.pdf(FIT_P_GRID)
            _metalog_fit = _mf
            _metalog_curve = (_xg, _pg)
            _metalog_modes = detect_modes_from_arrays(_xg, _pg)
        except MetalogError as e:
            _fit_error = f"Metalog: {e}"

        try:
            _constraint = ConstraintType[constraint_label]
            _qf = QFlex(x_sorted, y_plot_pos, terms=k_qflex_val, constraint_type=_constraint)
            _xg = _qf.quantile(FIT_P_GRID)
            _pg = _qf.pdf(FIT_P_GRID)
            _qflex_fit = _qf
            _qflex_curve = (_xg, _pg)
            _qflex_modes = detect_modes_from_arrays(_xg, _pg)
        except QFlexError as e:
            _fit_error = (_fit_error + " · " if _fit_error else "") + f"{QFLEX_LABELS[constraint_label]}: {e}"

        return {
            "fit_error": _fit_error,
            "metalog_fit": _metalog_fit, "metalog_curve": _metalog_curve, "metalog_modes": _metalog_modes,
            "qflex_fit": _qflex_fit, "qflex_curve": _qflex_curve, "qflex_modes": _qflex_modes,
        }

    def mode_summary_md(mo, fit_error, metalog_fit, metalog_modes, qflex_fit, qflex_modes, constraint_label):
        _qflex_name = QFLEX_LABELS[constraint_label]

        def _line(name, fit, modes):
            if fit is None:
                return f"**{name}:** fit failed"
            _n_modes = modes[0]
            _feas = "valid" if fit.is_feasible else "⚠️ infeasible (PDF goes negative)"
            _shape = (
                "unimodal" if _n_modes == 1
                else (f"**{_n_modes} modes — spurious structure**" if _n_modes and _n_modes > 1 else "no modes detected")
            )
            return f"**{name}:** {_feas}, {_shape}"

        return mo.md(
            f"{_line('Metalog', metalog_fit, metalog_modes)}  \n"
            f"{_line(_qflex_name, qflex_fit, qflex_modes)}"
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
                          qflex_modes, fit_error, x_range, y_range):
        """The QF + PDF panel pair shared by the Johnson MC section and the
        bimodal-mixture MC section: true curve, sample, both fits, both
        fits' true-curve overlay for contrast, and fixed axes."""
        _qflex_name = QFLEX_LABELS[constraint_label]
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
                        marker=dict(color="#4E5972", size=5, opacity=0.55)),
            row=1, col=1,
        )

        if metalog_curve is not None:
            _xg, _pg = metalog_curve
            _fig.add_trace(
                go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name=f"Metalog K={k_metalog_val}",
                            line=dict(color="#3B5FA0", width=2.4)),
                row=1, col=1,
            )
            _fig.add_trace(
                go.Scatter(x=_xg, y=_pg, mode="lines", name=f"Metalog K={k_metalog_val}",
                            line=dict(color="#3B5FA0", width=2.4), showlegend=False),
                row=1, col=2,
            )
            _n_modes, _locs, _hgts = metalog_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(
                    go.Scatter(x=_locs, y=_hgts, mode="markers", name="Metalog modes",
                                marker=dict(color="#3B5FA0", size=10, symbol="diamond",
                                            line=dict(color="white", width=1))),
                    row=1, col=2,
                )

        if qflex_curve is not None:
            _xg, _pg = qflex_curve
            _fig.add_trace(
                go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name=f"{_qflex_name} K={k_qflex_val}",
                            line=dict(color="#2E8B57", width=2.4)),
                row=1, col=1,
            )
            _fig.add_trace(
                go.Scatter(x=_xg, y=_pg, mode="lines", name=f"{_qflex_name} K={k_qflex_val}",
                            line=dict(color="#2E8B57", width=2.4), showlegend=False),
                row=1, col=2,
            )
            _n_modes, _locs, _hgts = qflex_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(
                    go.Scatter(x=_locs, y=_hgts, mode="markers", name=f"{_qflex_name} modes",
                                marker=dict(color="#2E8B57", size=10, symbol="diamond",
                                            line=dict(color="white", width=1))),
                    row=1, col=2,
                )

        _fig.update_xaxes(title_text="Cumulative probability", range=[0, 1], row=1, col=1)
        _fig.update_yaxes(title_text="Value", range=x_range, row=1, col=1)
        _fig.update_xaxes(title_text="Value", range=x_range, row=1, col=2)
        _fig.update_yaxes(title_text="Density", range=y_range, row=1, col=2)
        _fig.update_layout(
            height=430, margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(orientation="h", y=-0.18), uirevision="keep-zoom",
        )

        return mo.vstack([
            mo.ui.plotly(_fig, config=plotly_config),
            mode_summary_md(mo, fit_error, metalog_fit, metalog_modes, qflex_fit, qflex_modes, constraint_label),
        ])

    def render_empirical_panel(mo, plotly_config, title, axis_label, constraint_label, p_grid, eqf_point, eqf_lo,
                                 eqf_hi, x_raw, metalog_curve, metalog_fit, metalog_modes, qflex_curve, qflex_fit,
                                 qflex_modes, fit_error):
        """The EQF+CI / Metalog / QFlex panel pair shared by the three
        empirical dataset sections. The PDF panel also shows a histogram of
        the raw data for reference."""
        _qflex_name = QFLEX_LABELS[constraint_label]
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
        _p_raw = (np.arange(1, _n_raw + 1) - 0.3) / (_n_raw + 0.4)
        _stride = max(1, _n_raw // 400)
        _fig.add_trace(go.Scatter(
            x=_p_raw[::_stride], y=x_raw[::_stride], mode="markers", name="Raw data",
            marker=dict(color="#4E5972", size=4, opacity=0.35),
        ), row=1, col=1)

        _fig.add_trace(go.Histogram(
            x=x_raw, histnorm="probability density", name="Data histogram",
            marker=dict(color="#C7CCDA"), opacity=0.55, nbinsx=40,
        ), row=1, col=2)

        if metalog_curve is not None:
            _xg, _pg = metalog_curve
            _fig.add_trace(go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name="Metalog fit",
                                        line=dict(color="#3B5FA0", width=2.4)), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_xg, y=_pg, mode="lines", name="Metalog fit",
                                        line=dict(color="#3B5FA0", width=2.4), showlegend=False), row=1, col=2)
            _n_modes, _locs, _hgts = metalog_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(go.Scatter(x=_locs, y=_hgts, mode="markers", name="Metalog modes",
                                            marker=dict(color="#3B5FA0", size=10, symbol="diamond",
                                                        line=dict(color="white", width=1))), row=1, col=2)

        if qflex_curve is not None:
            _xg, _pg = qflex_curve
            _fig.add_trace(go.Scatter(x=FIT_P_GRID, y=_xg, mode="lines", name=f"{_qflex_name} fit",
                                        line=dict(color="#2E8B57", width=2.4)), row=1, col=1)
            _fig.add_trace(go.Scatter(x=_xg, y=_pg, mode="lines", name=f"{_qflex_name} fit",
                                        line=dict(color="#2E8B57", width=2.4), showlegend=False), row=1, col=2)
            _n_modes, _locs, _hgts = qflex_modes
            if _locs is not None and len(_locs) > 0:
                _fig.add_trace(go.Scatter(x=_locs, y=_hgts, mode="markers", name=f"{_qflex_name} modes",
                                            marker=dict(color="#2E8B57", size=10, symbol="diamond",
                                                        line=dict(color="white", width=1))), row=1, col=2)

        _fig.update_xaxes(title_text="Cumulative probability", range=[0, 1], row=1, col=1)
        _fig.update_yaxes(title_text=axis_label, row=1, col=1)
        _fig.update_xaxes(title_text=axis_label, row=1, col=2)
        _fig.update_yaxes(title_text="Density", row=1, col=2)
        _fig.update_layout(
            title=f"{title} — N={_n_raw}",
            height=430, margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=-0.18), dragmode="zoom", barmode="overlay",
        )

        return mo.vstack([
            mo.ui.plotly(_fig, config=plotly_config),
            mode_summary_md(mo, fit_error, metalog_fit, metalog_modes, qflex_fit, qflex_modes, constraint_label),
        ])

    def run_replicate_batch(mo, n_reps, k_metalog_val, k_qflex_val, constraint_label, draw_fn, seed,
                              metalog_w1_ref=None, qflex_w1_ref=None, w1_label="W1 vs true QF"):
        """Stream a batch of (fit Metalog + fit QFlex) replicates, updating
        the cell's output after every replicate, then leave a summary table
        behind -- feasibility rate, false-modality rate, and (when a
        reference curve is given) median W1 distance, the same statistics
        behind the paper's Table 2/3, scoped to whichever single scenario
        called this."""
        _qflex_name = QFLEX_LABELS[constraint_label]
        _constraint = ConstraintType[constraint_label]
        _dp = FIT_P_GRID[1] - FIT_P_GRID[0]
        _rng = np.random.default_rng(seed)
        _rows = []

        for _rep in range(n_reps):
            _x, _y = draw_fn(_rng)

            for _model_name, _fit_fn, _w1_ref in (
                ("Metalog", lambda: Metalog(_x, _y, terms=k_metalog_val), metalog_w1_ref),
                (_qflex_name, lambda: QFlex(_x, _y, terms=k_qflex_val, constraint_type=_constraint), qflex_w1_ref),
            ):
                _row = {"Replicate": _rep + 1, "Model": _model_name}
                try:
                    _fit = _fit_fn()
                    _xg = _fit.quantile(FIT_P_GRID)
                    _pg = _fit.pdf(FIT_P_GRID)
                    _n_modes, _, _ = detect_modes_from_arrays(_xg, _pg)
                    _row["Feasible"] = bool(_fit.is_feasible)
                    _row["Modes"] = int(_n_modes) if _n_modes is not None else 0
                    if _w1_ref is not None:
                        _row[w1_label] = round(float(np.sum(np.abs(_xg - _w1_ref)) * _dp), 4)
                except (MetalogError, QFlexError):
                    _row["Feasible"] = False
                    _row["Modes"] = None
                    if _w1_ref is not None:
                        _row[w1_label] = None
                _rows.append(_row)

            mo.output.replace(
                mo.vstack([
                    mo.md(f"Running replicate {_rep + 1} / {n_reps}..."),
                    mo.ui.table(_rows, selection=None, show_download=False, pagination=True, page_size=10),
                ])
            )

        _df = pd.DataFrame(_rows)
        _summary_rows = []
        for _model, _g in _df.groupby("Model", sort=False):
            _n = len(_g)
            _feas = _g[_g["Feasible"]]
            _feas_pct = round(100 * len(_feas) / _n) if _n else 0
            _false_modal_pct = round(100 * (_feas["Modes"] > 1).mean()) if len(_feas) else 0
            _summary = {"Model": _model, "Replicates": _n, "Feasibility %": _feas_pct, "False-modality %": _false_modal_pct}
            if w1_label in _g.columns:
                _summary[f"Median {w1_label}"] = round(_feas[w1_label].median(), 4) if len(_feas) else float("nan")
            _summary_rows.append(_summary)

        mo.output.replace(
            mo.vstack([
                mo.md(f"**Done — {n_reps} replicates.**"),
                mo.ui.table(_rows, selection=None, show_download=False, pagination=True, page_size=10),
                mo.md("**Summary across all replicates** &mdash; the same statistics behind the paper's Table 2/3."),
                mo.ui.table(_summary_rows, selection=None, show_download=False, pagination=False),
            ])
        )
        return

    return (
        FIT_P_GRID,
        QFLEX_LABELS,
        fit_metalog_qflex,
        mode_summary_md,
        render_empirical_panel,
        render_mc_panel,
        run_replicate_batch,
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
def _(mo):
    mo.md(
        r"""
        ---
        ## Johnson distributions

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
def _(PLOTLY_CONFIG, go, make_subplots, mo, np, sb_dist, sl_dist, su_dist):
    _fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Johnson SU (unbounded)", "Johnson SL (semi-bounded)", "Johnson SB (bounded)"),
        horizontal_spacing=0.06,
    )

    _p = np.linspace(0.002, 0.998, 500)
    _specs = [(su_dist, "#3B5FA0", 1), (sl_dist, "#2E8B57", 2), (sb_dist, "#A66A16", 3)]
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

    _fig.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    mo.ui.plotly(_fig, config=PLOTLY_CONFIG)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Monte Carlo & Bootstrap refit

        Pick which Johnson family (from the panel above) to treat as the
        true population, then redraw samples from it and watch Metalog and
        QFlex refit live &mdash; the mechanism the whole paper is about,
        made tangible. Axes are fixed to the true distribution's own range,
        so a K or N change never rescales the plot out from under you.
        """
    )
    return


@app.cell
def _(mo):
    family = mo.ui.dropdown(
        options=[
            "Johnson SU (unbounded)",
            "Johnson SL (semi-bounded)",
            "Johnson SB (bounded)",
        ],
        value="Johnson SU (unbounded)",
        label="Reference distribution (uses the η/κ/c/d panel above)",
    )
    sampling_mode = mo.ui.radio(
        options=[
            "Monte Carlo — fresh draw from the true distribution each time",
            "Bootstrap — resample one fixed realization each time",
        ],
        value="Monte Carlo — fresh draw from the true distribution each time",
        label="Sampling mode",
    )
    base_seed = mo.ui.number(
        start=0, stop=999_999, step=1, value=200612,
        label="Reference-sample seed (the \"one realization\" for bootstrap mode)",
    )
    redraw = mo.ui.button(
        label="🎲 Draw new sample",
        value=0, on_click=lambda v: v + 1,
    )
    new_reference = mo.ui.button(
        label="↻ New reference realization",
        value=0, on_click=lambda v: v + 1,
    )
    return base_seed, family, new_reference, redraw, sampling_mode


@app.cell
def _(mo):
    n_slider = mo.ui.slider(
        start=15, stop=500, step=5, value=200, label="Sample size N", show_value=True
    )
    k_metalog = mo.ui.slider(
        start=2, stop=15, step=1, value=9, label="Metalog K", show_value=True
    )
    k_qflex = mo.ui.slider(
        start=2, stop=15, step=1, value=9, label="QFlex K", show_value=True
    )
    qflex_constraint = mo.ui.dropdown(
        options={
            "Unconstrained": "NONE",
            "A+  (all coefficients ≥ 0)": "A",
            "TA+  (tail coefficients ≥ 0)": "TA",
        },
        value="Unconstrained",
        label="QFlex constraint",
    )
    return k_metalog, k_qflex, n_slider, qflex_constraint


@app.cell
def _(
    base_seed,
    family,
    k_metalog,
    k_qflex,
    mo,
    n_slider,
    new_reference,
    qflex_constraint,
    redraw,
    sampling_mode,
):
    mo.vstack(
        [
            mo.hstack([family, sampling_mode], justify="start", gap=2),
            mo.hstack([base_seed, new_reference, redraw], justify="start", gap=2),
            mo.md(
                f"*↻ New reference realization: **#{new_reference.value + 1}** &nbsp;&middot;&nbsp; "
                f"🎲 Draw new sample: **#{redraw.value + 1}***  \n"
                "*Each click reruns the fit — the counters above are the easiest way to confirm a "
                "click registered. \"New reference realization\" only changes anything in **Bootstrap** "
                "mode (it redraws the one fixed sample bootstrap resamples from); \"Draw new sample\" "
                "always redraws the sample shown in the plot below.*"
            ),
            mo.hstack([n_slider, k_metalog, k_qflex], justify="start", gap=2),
            qflex_constraint,
        ],
        gap=1,
    )
    return


@app.cell
def _(k_metalog, k_qflex, mo, n_slider):
    # Feature 04 — playground sample size, with a K-aware floor: a fit needs at
    # least K distinct points, so N can never be allowed to drop below the
    # larger of the two chosen K's, or the fit crashes instead of failing
    # gracefully.
    _floor = max(k_metalog.value, k_qflex.value, 8)
    n_effective = max(n_slider.value, _floor)

    n_floor_notice = (
        mo.callout(
            f"N={n_slider.value} is below the current max K ({_floor}); "
            f"using N={n_effective} so both fits stay solvable.",
            kind="warn",
        )
        if n_slider.value < _floor
        else None
    )
    n_floor_notice
    return (n_effective,)


@app.cell
def _(family, sb_dist, sl_dist, su_dist):
    _dist_by_name = {
        "Johnson SU (unbounded)": su_dist,
        "Johnson SL (semi-bounded)": sl_dist,
        "Johnson SB (bounded)": sb_dist,
    }
    true_dist = _dist_by_name[family.value]
    return (true_dist,)


@app.cell
def _(base_seed, new_reference, np, n_effective, true_dist):
    # The "single selected realization" bootstrap resampling is conditioned on.
    # Regenerated when the reference seed changes, the "new reference
    # realization" button is clicked, or N changes (so it always matches the
    # current sample-size setting).
    _rng = np.random.default_rng(base_seed.value + new_reference.value)
    reference_sample = np.sort(true_dist.quantile(_rng.random(n_effective)))
    return (reference_sample,)


@app.cell
def _(np, n_effective, reference_sample, redraw, sampling_mode, true_dist):
    # The actual draw shown in the plot below: a fresh IID Monte Carlo sample,
    # or a bootstrap resample of the single reference realization above.
    _rng = np.random.default_rng(10_000 + redraw.value)
    if sampling_mode.value.startswith("Monte Carlo"):
        x_sample = np.sort(true_dist.quantile(_rng.random(n_effective)))
    else:
        x_sample = np.sort(_rng.choice(reference_sample, size=n_effective, replace=True))

    _n = len(x_sample)
    y_sample = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)  # Weibull-type plotting position
    return x_sample, y_sample


@app.cell
def _(fit_metalog_qflex, k_metalog, k_qflex, qflex_constraint, x_sample, y_sample):
    _res = fit_metalog_qflex(x_sample, y_sample, k_metalog.value, k_qflex.value, qflex_constraint.value)
    fit_error = _res["fit_error"]
    metalog_fit, metalog_curve, metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    qflex_fit, qflex_curve, qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return fit_error, metalog_curve, metalog_fit, metalog_modes, qflex_curve, qflex_fit, qflex_modes


@app.cell
def _(true_dist, true_dist_ranges):
    mc_x_range, mc_y_range = true_dist_ranges(true_dist)
    return mc_x_range, mc_y_range


@app.cell
def _(
    PLOTLY_CONFIG,
    fit_error,
    k_metalog,
    k_qflex,
    mc_x_range,
    mc_y_range,
    metalog_curve,
    metalog_fit,
    metalog_modes,
    mo,
    qflex_constraint,
    qflex_curve,
    qflex_fit,
    qflex_modes,
    render_mc_panel,
    true_dist,
    x_sample,
    y_sample,
):
    render_mc_panel(
        mo, PLOTLY_CONFIG, true_dist, x_sample, y_sample, k_metalog.value, k_qflex.value, qflex_constraint.value,
        metalog_curve, metalog_fit, metalog_modes, qflex_curve, qflex_fit, qflex_modes,
        fit_error, mc_x_range, mc_y_range,
    )
    return


@app.cell
def _(mo):
    mc_n_replicates = mo.ui.slider(
        start=5, stop=100, step=5, value=30, label="Replicates", show_value=True
    )
    mc_run_batch = mo.ui.run_button(label="▶ Run batch for this scenario")
    mo.md("**Full simulation for this scenario** — same family, sampling mode, N, and K's as above.")
    mo.hstack([mc_n_replicates, mc_run_batch], justify="start", gap=2)
    return mc_n_replicates, mc_run_batch


@app.cell
def _(
    FIT_P_GRID,
    base_seed,
    k_metalog,
    k_qflex,
    mc_n_replicates,
    mc_run_batch,
    mo,
    n_effective,
    np,
    qflex_constraint,
    reference_sample,
    run_replicate_batch,
    sampling_mode,
    true_dist,
):
    if mc_run_batch.value:
        _x_true_grid = true_dist.quantile(FIT_P_GRID)

        def _draw(rng):
            if sampling_mode.value.startswith("Monte Carlo"):
                _x = np.sort(true_dist.quantile(rng.random(n_effective)))
            else:
                _x = np.sort(rng.choice(reference_sample, size=n_effective, replace=True))
            _n = len(_x)
            _y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
            return _x, _y

        run_replicate_batch(
            mo, mc_n_replicates.value, k_metalog.value, k_qflex.value, qflex_constraint.value,
            _draw, base_seed.value + 777, metalog_w1_ref=_x_true_grid, qflex_w1_ref=_x_true_grid,
        )
    else:
        mo.output.replace(
            mo.md("*Click **▶ Run batch for this scenario** to simulate replicates with the current controls.*")
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Bimodal mixture playground

        A two-component mixture built from the **Johnson SU panel above**:
        both components share that panel's shape, offset from each other by
        a controllable distance, and combined with a controllable mixture
        weight. This is the paper's motivating case made explorable &mdash;
        watch Metalog/QFlex either recover the real second mode or invent
        one that isn't there.
        """
    )
    return


@app.cell
def _(mo):
    delta_sd = mo.ui.slider(
        start=0.0, stop=6.0, step=0.1, value=2.5,
        label="Mean separation (in SDs of the base SU)", show_value=True,
    )
    mixture_ratio = mo.ui.slider(
        start=0.05, stop=0.95, step=0.05, value=0.5,
        label="Mixture weight (share on component A)", show_value=True,
    )
    return delta_sd, mixture_ratio


@app.cell
def _(mo):
    bimodal_n_slider = mo.ui.slider(
        start=15, stop=500, step=5, value=200, label="Sample size N", show_value=True
    )
    bimodal_k_metalog = mo.ui.slider(
        start=2, stop=15, step=1, value=9, label="Metalog K", show_value=True
    )
    bimodal_k_qflex = mo.ui.slider(
        start=2, stop=15, step=1, value=9, label="QFlex K", show_value=True
    )
    bimodal_qflex_constraint = mo.ui.dropdown(
        options={
            "Unconstrained": "NONE",
            "A+  (all coefficients ≥ 0)": "A",
            "TA+  (tail coefficients ≥ 0)": "TA",
        },
        value="Unconstrained",
        label="QFlex constraint",
    )
    return bimodal_k_metalog, bimodal_k_qflex, bimodal_n_slider, bimodal_qflex_constraint


@app.cell
def _(mo):
    bimodal_sampling_mode = mo.ui.radio(
        options=[
            "Monte Carlo — fresh draw from the true mixture each time",
            "Bootstrap — resample one fixed realization each time",
        ],
        value="Monte Carlo — fresh draw from the true mixture each time",
        label="Sampling mode",
    )
    bimodal_seed = mo.ui.number(
        start=0, stop=999_999, step=1, value=531_204,
        label="Reference-sample seed (bootstrap mode's fixed realization)",
    )
    bimodal_redraw = mo.ui.button(label="🎲 Draw new sample", value=0, on_click=lambda v: v + 1)
    bimodal_new_reference = mo.ui.button(
        label="↻ New reference realization", value=0, on_click=lambda v: v + 1
    )
    return bimodal_new_reference, bimodal_redraw, bimodal_sampling_mode, bimodal_seed


@app.cell
def _(
    bimodal_k_metalog,
    bimodal_k_qflex,
    bimodal_n_slider,
    bimodal_new_reference,
    bimodal_qflex_constraint,
    bimodal_redraw,
    bimodal_sampling_mode,
    bimodal_seed,
    delta_sd,
    mixture_ratio,
    mo,
):
    mo.vstack(
        [
            mo.hstack([delta_sd, mixture_ratio], justify="start", gap=2),
            mo.hstack([bimodal_n_slider, bimodal_k_metalog, bimodal_k_qflex], justify="start", gap=2),
            bimodal_qflex_constraint,
            mo.hstack([bimodal_sampling_mode], justify="start", gap=2),
            mo.hstack([bimodal_seed, bimodal_new_reference, bimodal_redraw], justify="start", gap=2),
            mo.md(
                f"*↻ New reference realization: **#{bimodal_new_reference.value + 1}** &nbsp;&middot;&nbsp; "
                f"🎲 Draw new sample: **#{bimodal_redraw.value + 1}***"
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(bimodal_k_metalog, bimodal_k_qflex, bimodal_n_slider, mo):
    _floor = max(bimodal_k_metalog.value, bimodal_k_qflex.value, 8)
    bimodal_n_effective = max(bimodal_n_slider.value, _floor)

    _notice = (
        mo.callout(
            f"N={bimodal_n_slider.value} is below the current max K ({_floor}); "
            f"using N={bimodal_n_effective} so both fits stay solvable.",
            kind="warn",
        )
        if bimodal_n_slider.value < _floor
        else None
    )
    _notice
    return (bimodal_n_effective,)


@app.cell
def _(delta_sd, mixture_ratio, np, su_dist):
    # Component A is exactly the Johnson SU panel above. Component B is the
    # same shape, shifted by delta_sd * sigma, where sigma is su_dist's own
    # standard deviation, estimated numerically off a fine quantile grid
    # (Johnson quantile functions have no closed-form variance).
    _p_grid = np.linspace(0.0005, 0.9995, 5000)
    _su_sigma = float(np.std(su_dist.quantile(_p_grid)))

    class _ShiftedSU:
        """su_dist translated by a fixed offset -- same shape, new eta."""
        def __init__(self, base, offset):
            self._base = base
            self._offset = offset

        def quantile(self, p):
            return self._base.quantile(p) + self._offset

        def pdf(self, x):
            return self._base.pdf(np.asarray(x) - self._offset)

    bimodal_comp_a = su_dist
    bimodal_comp_b = _ShiftedSU(su_dist, delta_sd.value * _su_sigma)
    bimodal_weight_a = mixture_ratio.value
    return bimodal_comp_a, bimodal_comp_b, bimodal_weight_a


@app.cell
def _(bimodal_comp_a, bimodal_comp_b, bimodal_weight_a, np):
    class _MixtureDist:
        """Numeric PDF + quantile for the mixture, used only for the "true
        curve" overlay -- actual sampling below inverts each component's
        exact quantile function directly, not this numeric approximation."""
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
            _cdf = _cdf / _cdf[-1]
            self._cdf_grid = _cdf

        def pdf(self, x):
            return np.interp(x, self._x_grid, self._pdf_grid, left=0.0, right=0.0)

        def quantile(self, p):
            return np.interp(np.asarray(p), self._cdf_grid, self._x_grid)

    bimodal_dist = _MixtureDist(bimodal_comp_a, bimodal_comp_b, bimodal_weight_a)
    return (bimodal_dist,)


@app.cell
def _(bimodal_comp_a, bimodal_comp_b, bimodal_n_effective, bimodal_new_reference, bimodal_seed, bimodal_weight_a, np):
    _rng = np.random.default_rng(bimodal_seed.value + bimodal_new_reference.value)
    _which_a = _rng.random(bimodal_n_effective) < bimodal_weight_a
    _u = _rng.random(bimodal_n_effective)
    _vals = np.where(_which_a, bimodal_comp_a.quantile(_u), bimodal_comp_b.quantile(_u))
    bimodal_reference_sample = np.sort(_vals)
    return (bimodal_reference_sample,)


@app.cell
def _(
    bimodal_comp_a,
    bimodal_comp_b,
    bimodal_n_effective,
    bimodal_reference_sample,
    bimodal_redraw,
    bimodal_sampling_mode,
    bimodal_weight_a,
    np,
):
    _rng = np.random.default_rng(20_000 + bimodal_redraw.value)
    if bimodal_sampling_mode.value.startswith("Monte Carlo"):
        _which_a = _rng.random(bimodal_n_effective) < bimodal_weight_a
        _u = _rng.random(bimodal_n_effective)
        _vals = np.where(_which_a, bimodal_comp_a.quantile(_u), bimodal_comp_b.quantile(_u))
        bimodal_x_sample = np.sort(_vals)
    else:
        bimodal_x_sample = np.sort(_rng.choice(bimodal_reference_sample, size=bimodal_n_effective, replace=True))

    _n = len(bimodal_x_sample)
    bimodal_y_sample = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
    return bimodal_x_sample, bimodal_y_sample


@app.cell
def _(
    bimodal_k_metalog,
    bimodal_k_qflex,
    bimodal_qflex_constraint,
    bimodal_x_sample,
    bimodal_y_sample,
    fit_metalog_qflex,
):
    _res = fit_metalog_qflex(
        bimodal_x_sample, bimodal_y_sample, bimodal_k_metalog.value, bimodal_k_qflex.value,
        bimodal_qflex_constraint.value,
    )
    bimodal_fit_error = _res["fit_error"]
    bimodal_metalog_fit, bimodal_metalog_curve, bimodal_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    bimodal_qflex_fit, bimodal_qflex_curve, bimodal_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return bimodal_fit_error, bimodal_metalog_curve, bimodal_metalog_fit, bimodal_metalog_modes, bimodal_qflex_curve, bimodal_qflex_fit, bimodal_qflex_modes


@app.cell
def _(bimodal_dist, true_dist_ranges):
    bimodal_x_range, bimodal_y_range = true_dist_ranges(bimodal_dist, p_lo=0.0005, p_hi=0.9995)
    return bimodal_x_range, bimodal_y_range


@app.cell
def _(
    PLOTLY_CONFIG,
    bimodal_dist,
    bimodal_fit_error,
    bimodal_k_metalog,
    bimodal_k_qflex,
    bimodal_metalog_curve,
    bimodal_metalog_fit,
    bimodal_metalog_modes,
    bimodal_qflex_constraint,
    bimodal_qflex_curve,
    bimodal_qflex_fit,
    bimodal_qflex_modes,
    bimodal_x_range,
    bimodal_x_sample,
    bimodal_y_range,
    bimodal_y_sample,
    mo,
    render_mc_panel,
):
    render_mc_panel(
        mo, PLOTLY_CONFIG, bimodal_dist, bimodal_x_sample, bimodal_y_sample, bimodal_k_metalog.value,
        bimodal_k_qflex.value, bimodal_qflex_constraint.value, bimodal_metalog_curve, bimodal_metalog_fit,
        bimodal_metalog_modes, bimodal_qflex_curve, bimodal_qflex_fit, bimodal_qflex_modes,
        bimodal_fit_error, bimodal_x_range, bimodal_y_range,
    )
    return


@app.cell
def _(mo):
    bimodal_n_replicates = mo.ui.slider(
        start=5, stop=100, step=5, value=30, label="Replicates", show_value=True
    )
    bimodal_run_batch = mo.ui.run_button(label="▶ Run batch for this scenario")
    mo.md("**Full simulation for this scenario** — same mixture, sampling mode, N, and K's as above.")
    mo.hstack([bimodal_n_replicates, bimodal_run_batch], justify="start", gap=2)
    return bimodal_n_replicates, bimodal_run_batch


@app.cell
def _(
    FIT_P_GRID,
    bimodal_dist,
    bimodal_k_metalog,
    bimodal_k_qflex,
    bimodal_n_effective,
    bimodal_n_replicates,
    bimodal_qflex_constraint,
    bimodal_reference_sample,
    bimodal_run_batch,
    bimodal_sampling_mode,
    bimodal_seed,
    mo,
    np,
    run_replicate_batch,
):
    if bimodal_run_batch.value:
        _x_true_grid = bimodal_dist.quantile(FIT_P_GRID)

        def _draw(rng):
            if bimodal_sampling_mode.value.startswith("Monte Carlo"):
                _x = np.sort(bimodal_dist.quantile(rng.random(bimodal_n_effective)))
            else:
                _x = np.sort(rng.choice(bimodal_reference_sample, size=bimodal_n_effective, replace=True))
            _n = len(_x)
            _y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
            return _x, _y

        run_replicate_batch(
            mo, bimodal_n_replicates.value, bimodal_k_metalog.value, bimodal_k_qflex.value,
            bimodal_qflex_constraint.value, _draw, bimodal_seed.value + 777,
            metalog_w1_ref=_x_true_grid, qflex_w1_ref=_x_true_grid,
        )
    else:
        mo.output.replace(
            mo.md("*Click **▶ Run batch for this scenario** to simulate replicates with the current controls.*")
        )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Empirical case studies

        The paper's three real datasets, each in its own section with its
        **own** Metalog K / QFlex K / constraint controls and its own
        per-scenario batch simulation. Every panel shows the empirical
        quantile function (raw sorted data as a step-like curve) with a
        pointwise 95% bootstrap CI, plus a Metalog/QFlex fit, plus (on the
        density panel) a histogram of the raw data for reference.
        **Drag a rectangle on a plot to zoom**; double-click to reset.
        """
    )
    return


@app.cell
def _(DATA_DIR, np, pd):
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
        # an https:// URL in the deployed WASM build -- pandas' read_csv
        # (via the pyodide-http patch loaded alongside it) transparently
        # fetches either; a plain open() cannot follow the URL form, so this
        # must go through pandas like the other two loaders.
        _df = pd.read_csv(
            str(DATA_DIR / "geyser.txt"), sep=r"\s+", header=None,
            names=["eruption", "waiting", "indicator"],
        )
        return np.sort(_df["waiting"].dropna().values.astype(float))

    def eqf_bootstrap_ci(x_raw, p_grid, n_boot, seed=42):
        # Pointwise 95% percentile bootstrap CI on the EQF, matching the
        # method used for the paper's own EQF+CI figures (e.g. Figure 10,
        # hydrology): resample the raw data with replacement, interpolate
        # each resample's EQF onto a common probability grid, and take the
        # 2.5th/97.5th percentiles at each grid point across resamples.
        _N = len(x_raw)
        _p_emp = np.arange(1, _N + 1) / (_N + 1)
        _rng = np.random.default_rng(seed)
        _boot = np.empty((n_boot, len(p_grid)))
        for _b in range(n_boot):
            _x_boot = np.sort(_rng.choice(x_raw, size=_N, replace=True))
            _boot[_b] = np.interp(p_grid, _p_emp, _x_boot)
        _point = np.interp(p_grid, _p_emp, x_raw)
        _lo = np.percentile(_boot, 2.5, axis=0)
        _hi = np.percentile(_boot, 97.5, axis=0)
        return _point, _lo, _hi

    return eqf_bootstrap_ci, load_fish_raw, load_geyser_raw, load_hydrology_raw


@app.cell
def _(mo):
    mo.md(r"""### Fish weights (`Fish Biology.xlsx`)""")
    return


@app.cell
def _(mo):
    fish_jitter = mo.ui.slider(
        start=0.0, stop=0.5, step=0.05, value=0.25,
        label="Jitter width (lbs)", show_value=True,
    )
    mo.vstack([
        fish_jitter,
        mo.md(
            "*Fish weights were recorded to the nearest 0.5 lb, so the raw "
            "data has heavy ties (only ~56 distinct values across 3,474 "
            "observations). This slider adds `Uniform(-width/2, +width/2)` "
            "noise to break those ties before fitting — width 0 uses the "
            "raw, unjittered data.*"
        ),
    ])
    return (fish_jitter,)


@app.cell
def _(mo):
    fish_k_metalog = mo.ui.slider(start=2, stop=15, step=1, value=9, label="Metalog K", show_value=True)
    fish_k_qflex = mo.ui.slider(start=2, stop=15, step=1, value=9, label="QFlex K", show_value=True)
    fish_qflex_constraint = mo.ui.dropdown(
        options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
        value="Unconstrained", label="QFlex constraint",
    )
    mo.vstack([mo.hstack([fish_k_metalog, fish_k_qflex], justify="start", gap=2), fish_qflex_constraint])
    return fish_k_metalog, fish_k_qflex, fish_qflex_constraint


@app.cell
def _(fish_jitter, load_fish_raw, np):
    _raw = load_fish_raw()
    if fish_jitter.value > 0:
        _rng = np.random.default_rng(20260828)
        _jittered = _raw + _rng.uniform(-fish_jitter.value / 2, fish_jitter.value / 2, size=len(_raw))
        fish_x = np.sort(_jittered)
    else:
        fish_x = _raw
    _n = len(fish_x)
    fish_y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
    return fish_x, fish_y


@app.cell
def _(eqf_bootstrap_ci, fish_x, np):
    fish_p_grid = np.linspace(0.01, 0.99, 300)
    fish_eqf_point, fish_eqf_lo, fish_eqf_hi = eqf_bootstrap_ci(fish_x, fish_p_grid, n_boot=300, seed=42)
    return fish_eqf_hi, fish_eqf_lo, fish_eqf_point, fish_p_grid


@app.cell
def _(fish_k_metalog, fish_k_qflex, fish_qflex_constraint, fish_x, fish_y, fit_metalog_qflex):
    _res = fit_metalog_qflex(fish_x, fish_y, fish_k_metalog.value, fish_k_qflex.value, fish_qflex_constraint.value)
    fish_fit_error = _res["fit_error"]
    fish_metalog_fit, fish_metalog_curve, fish_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    fish_qflex_fit, fish_qflex_curve, fish_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return fish_fit_error, fish_metalog_curve, fish_metalog_fit, fish_metalog_modes, fish_qflex_curve, fish_qflex_fit, fish_qflex_modes


@app.cell
def _(
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
        fish_qflex_fit, fish_qflex_modes, fish_fit_error,
    )
    return


@app.cell
def _(mo):
    fish_n_replicates = mo.ui.slider(start=5, stop=100, step=5, value=30, label="Replicates", show_value=True)
    fish_run_batch = mo.ui.run_button(label="▶ Run batch for this dataset")
    mo.md("**Full simulation for this dataset** — bootstrap-resample the (jittered) fish weights and refit repeatedly.")
    mo.hstack([fish_n_replicates, fish_run_batch], justify="start", gap=2)
    return fish_n_replicates, fish_run_batch


@app.cell
def _(
    fish_k_metalog,
    fish_k_qflex,
    fish_metalog_curve,
    fish_n_replicates,
    fish_qflex_constraint,
    fish_qflex_curve,
    fish_run_batch,
    fish_x,
    mo,
    np,
    run_replicate_batch,
):
    if fish_run_batch.value:
        _metalog_ref = fish_metalog_curve[0] if fish_metalog_curve is not None else None
        _qflex_ref = fish_qflex_curve[0] if fish_qflex_curve is not None else None

        def _draw(rng):
            _n = len(fish_x)
            _x = np.sort(rng.choice(fish_x, size=_n, replace=True))
            _y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
            return _x, _y

        run_replicate_batch(
            mo, fish_n_replicates.value, fish_k_metalog.value, fish_k_qflex.value, fish_qflex_constraint.value,
            _draw, 471_001, metalog_w1_ref=_metalog_ref, qflex_w1_ref=_qflex_ref,
            w1_label="W1 vs full-sample fit",
        )
    else:
        mo.output.replace(mo.md("*Click **▶ Run batch for this dataset** to bootstrap-resample and refit repeatedly.*"))
    return


@app.cell
def _(mo):
    mo.md(r"""### River gauge height (`Hydrology.xlsx`)""")
    return


@app.cell
def _(mo):
    hydro_k_metalog = mo.ui.slider(start=2, stop=15, step=1, value=9, label="Metalog K", show_value=True)
    hydro_k_qflex = mo.ui.slider(start=2, stop=15, step=1, value=9, label="QFlex K", show_value=True)
    hydro_qflex_constraint = mo.ui.dropdown(
        options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
        value="Unconstrained", label="QFlex constraint",
    )
    mo.vstack([mo.hstack([hydro_k_metalog, hydro_k_qflex], justify="start", gap=2), hydro_qflex_constraint])
    return hydro_k_metalog, hydro_k_qflex, hydro_qflex_constraint


@app.cell
def _(load_hydrology_raw, np):
    hydro_x = load_hydrology_raw()
    _n = len(hydro_x)
    hydro_y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
    return hydro_x, hydro_y


@app.cell
def _(eqf_bootstrap_ci, hydro_x, np):
    hydro_p_grid = np.linspace(0.01, 0.99, 300)
    hydro_eqf_point, hydro_eqf_lo, hydro_eqf_hi = eqf_bootstrap_ci(hydro_x, hydro_p_grid, n_boot=300, seed=42)
    return hydro_eqf_hi, hydro_eqf_lo, hydro_eqf_point, hydro_p_grid


@app.cell
def _(fit_metalog_qflex, hydro_k_metalog, hydro_k_qflex, hydro_qflex_constraint, hydro_x, hydro_y):
    _res = fit_metalog_qflex(hydro_x, hydro_y, hydro_k_metalog.value, hydro_k_qflex.value, hydro_qflex_constraint.value)
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
    )
    return


@app.cell
def _(mo):
    hydro_n_replicates = mo.ui.slider(start=5, stop=100, step=5, value=30, label="Replicates", show_value=True)
    hydro_run_batch = mo.ui.run_button(label="▶ Run batch for this dataset")
    mo.md("**Full simulation for this dataset** — bootstrap-resample the gauge-height data and refit repeatedly.")
    mo.hstack([hydro_n_replicates, hydro_run_batch], justify="start", gap=2)
    return hydro_n_replicates, hydro_run_batch


@app.cell
def _(
    hydro_k_metalog,
    hydro_k_qflex,
    hydro_metalog_curve,
    hydro_n_replicates,
    hydro_qflex_constraint,
    hydro_qflex_curve,
    hydro_run_batch,
    hydro_x,
    mo,
    np,
    run_replicate_batch,
):
    if hydro_run_batch.value:
        _metalog_ref = hydro_metalog_curve[0] if hydro_metalog_curve is not None else None
        _qflex_ref = hydro_qflex_curve[0] if hydro_qflex_curve is not None else None

        def _draw(rng):
            _n = len(hydro_x)
            _x = np.sort(rng.choice(hydro_x, size=_n, replace=True))
            _y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
            return _x, _y

        run_replicate_batch(
            mo, hydro_n_replicates.value, hydro_k_metalog.value, hydro_k_qflex.value, hydro_qflex_constraint.value,
            _draw, 471_002, metalog_w1_ref=_metalog_ref, qflex_w1_ref=_qflex_ref,
            w1_label="W1 vs full-sample fit",
        )
    else:
        mo.output.replace(mo.md("*Click **▶ Run batch for this dataset** to bootstrap-resample and refit repeatedly.*"))
    return


@app.cell
def _(mo):
    mo.md(r"""### Old Faithful waiting time (`geyser.txt`)""")
    return


@app.cell
def _(mo):
    geyser_k_metalog = mo.ui.slider(start=2, stop=15, step=1, value=9, label="Metalog K", show_value=True)
    geyser_k_qflex = mo.ui.slider(start=2, stop=15, step=1, value=9, label="QFlex K", show_value=True)
    geyser_qflex_constraint = mo.ui.dropdown(
        options={"Unconstrained": "NONE", "A+  (all coefficients ≥ 0)": "A", "TA+  (tail coefficients ≥ 0)": "TA"},
        value="Unconstrained", label="QFlex constraint",
    )
    mo.vstack([mo.hstack([geyser_k_metalog, geyser_k_qflex], justify="start", gap=2), geyser_qflex_constraint])
    return geyser_k_metalog, geyser_k_qflex, geyser_qflex_constraint


@app.cell
def _(load_geyser_raw, np):
    geyser_x = load_geyser_raw()
    _n = len(geyser_x)
    geyser_y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
    return geyser_x, geyser_y


@app.cell
def _(eqf_bootstrap_ci, geyser_x, np):
    geyser_p_grid = np.linspace(0.01, 0.99, 300)
    geyser_eqf_point, geyser_eqf_lo, geyser_eqf_hi = eqf_bootstrap_ci(geyser_x, geyser_p_grid, n_boot=300, seed=42)
    return geyser_eqf_hi, geyser_eqf_lo, geyser_eqf_point, geyser_p_grid


@app.cell
def _(fit_metalog_qflex, geyser_k_metalog, geyser_k_qflex, geyser_qflex_constraint, geyser_x, geyser_y):
    _res = fit_metalog_qflex(
        geyser_x, geyser_y, geyser_k_metalog.value, geyser_k_qflex.value, geyser_qflex_constraint.value
    )
    geyser_fit_error = _res["fit_error"]
    geyser_metalog_fit, geyser_metalog_curve, geyser_metalog_modes = _res["metalog_fit"], _res["metalog_curve"], _res["metalog_modes"]
    geyser_qflex_fit, geyser_qflex_curve, geyser_qflex_modes = _res["qflex_fit"], _res["qflex_curve"], _res["qflex_modes"]
    return geyser_fit_error, geyser_metalog_curve, geyser_metalog_fit, geyser_metalog_modes, geyser_qflex_curve, geyser_qflex_fit, geyser_qflex_modes


@app.cell
def _(
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
        geyser_fit_error,
    )
    return


@app.cell
def _(mo):
    geyser_n_replicates = mo.ui.slider(start=5, stop=100, step=5, value=30, label="Replicates", show_value=True)
    geyser_run_batch = mo.ui.run_button(label="▶ Run batch for this dataset")
    mo.md("**Full simulation for this dataset** — bootstrap-resample the waiting times and refit repeatedly.")
    mo.hstack([geyser_n_replicates, geyser_run_batch], justify="start", gap=2)
    return geyser_n_replicates, geyser_run_batch


@app.cell
def _(
    geyser_k_metalog,
    geyser_k_qflex,
    geyser_metalog_curve,
    geyser_n_replicates,
    geyser_qflex_constraint,
    geyser_qflex_curve,
    geyser_run_batch,
    geyser_x,
    mo,
    np,
    run_replicate_batch,
):
    if geyser_run_batch.value:
        _metalog_ref = geyser_metalog_curve[0] if geyser_metalog_curve is not None else None
        _qflex_ref = geyser_qflex_curve[0] if geyser_qflex_curve is not None else None

        def _draw(rng):
            _n = len(geyser_x)
            _x = np.sort(rng.choice(geyser_x, size=_n, replace=True))
            _y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)
            return _x, _y

        run_replicate_batch(
            mo, geyser_n_replicates.value, geyser_k_metalog.value, geyser_k_qflex.value,
            geyser_qflex_constraint.value, _draw, 471_003,
            metalog_w1_ref=_metalog_ref, qflex_w1_ref=_qflex_ref, w1_label="W1 vs full-sample fit",
        )
    else:
        mo.output.replace(mo.md("*Click **▶ Run batch for this dataset** to bootstrap-resample and refit repeatedly.*"))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        *Built as a companion to Khanna &amp; Bickel, "Inferring Distributional
        Features based on Quantile-Parameterized Distribution Fits."*
        """
    )
    return


if __name__ == "__main__":
    app.run()
