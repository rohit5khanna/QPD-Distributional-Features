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
        Quantile-Parameterized Distribution Fits* (Khanna & Bickel). Redraw a
        sample below and watch the Metalog and QFlex fits move &mdash;
        including the spurious modes that motivate the paper.

        **Status:** 🟢 Live redraw & refit (this panel) · 🟢 free sample-size
        playground · 🟢 empirical case studies · 🟢 full-simulation batch mode.
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
        label="Reference distribution",
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
    redraw = mo.ui.button(label="🎲 Draw new sample", value=0, on_click=lambda v: v + 1)
    new_reference = mo.ui.button(
        label="↻ New reference realization", value=0, on_click=lambda v: v + 1
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
def _(JohnsonSB, JohnsonSL, JohnsonSU, family):
    # Fixed reference-distribution parameters, matching Bickel (2026) Fig. 1 /
    # the paper's Monte Carlo study — only the *family* is playground-adjustable
    # for now; exposing the shape parameters themselves is one of the open
    # decisions noted in the planning doc.
    _dist_builders = {
        "Johnson SU (unbounded)": lambda: JohnsonSU(eta=0.0, kappa=1.0, c=0.5, d=1.2),
        "Johnson SL (semi-bounded)": lambda: JohnsonSL(eta=0.0, kappa=1.0, c=0.0, d=1.0),
        "Johnson SB (bounded)": lambda: JohnsonSB(eta=0.0, kappa=1.0, c=0.0, d=1.0),
    }
    true_dist = _dist_builders[family.value]()
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
def _(
    ConstraintType,
    Metalog,
    MetalogError,
    QFlex,
    QFlexError,
    detect_modes_from_arrays,
    k_metalog,
    k_qflex,
    np,
    qflex_constraint,
    x_sample,
    y_sample,
):
    _ygrid = np.linspace(0.01, 0.99, 1000)

    fit_error = None
    metalog_fit = qflex_fit = None
    metalog_curve = qflex_curve = None
    metalog_modes = qflex_modes = (None, None, None)

    try:
        metalog_fit = Metalog(x_sample, y_sample, terms=k_metalog.value)
        _xg = metalog_fit.quantile(_ygrid)
        _pg = metalog_fit.pdf(_ygrid)
        metalog_curve = (_xg, _pg)
        metalog_modes = detect_modes_from_arrays(_xg, _pg)
    except MetalogError as e:
        fit_error = f"Metalog: {e}"

    try:
        _constraint = ConstraintType[qflex_constraint.value]
        qflex_fit = QFlex(x_sample, y_sample, terms=k_qflex.value, constraint_type=_constraint)
        _xg = qflex_fit.quantile(_ygrid)
        _pg = qflex_fit.pdf(_ygrid)
        qflex_curve = (_xg, _pg)
        qflex_modes = detect_modes_from_arrays(_xg, _pg)
    except QFlexError as e:
        fit_error = (fit_error + " · " if fit_error else "") + f"QFlex: {e}"
    return fit_error, metalog_curve, metalog_fit, metalog_modes, qflex_curve, qflex_fit, qflex_modes


@app.cell
def _(
    fit_error,
    k_metalog,
    k_qflex,
    metalog_curve,
    metalog_fit,
    metalog_modes,
    mo,
    qflex_curve,
    qflex_fit,
    qflex_modes,
    x_sample,
    y_sample,
    true_dist,
    go,
    make_subplots,
    np,
):
    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Quantile function", "Probability density"),
        horizontal_spacing=0.09,
    )

    _p_true = np.linspace(0.005, 0.995, 400)
    _x_true = true_dist.quantile(_p_true)
    _fig.add_trace(
        go.Scatter(x=_p_true, y=_x_true, mode="lines", name="True QF",
                    line=dict(color="#9AA3B8", width=1.6, dash="dot")),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scatter(x=y_sample, y=x_sample, mode="markers", name="Sample",
                    marker=dict(color="#4E5972", size=5, opacity=0.55)),
        row=1, col=1,
    )

    if metalog_curve is not None:
        _xg, _pg = metalog_curve
        _fig.add_trace(
            go.Scatter(x=np.linspace(0.01, 0.99, 1000), y=_xg, mode="lines",
                        name=f"Metalog K={k_metalog.value}",
                        line=dict(color="#3B5FA0", width=2.4)),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=_xg, y=_pg, mode="lines", name=f"Metalog K={k_metalog.value}",
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
            go.Scatter(x=np.linspace(0.01, 0.99, 1000), y=_xg, mode="lines",
                        name=f"QFlex K={k_qflex.value}",
                        line=dict(color="#2E8B57", width=2.4)),
            row=1, col=1,
        )
        _fig.add_trace(
            go.Scatter(x=_xg, y=_pg, mode="lines", name=f"QFlex K={k_qflex.value}",
                        line=dict(color="#2E8B57", width=2.4), showlegend=False),
            row=1, col=2,
        )
        _n_modes, _locs, _hgts = qflex_modes
        if _locs is not None and len(_locs) > 0:
            _fig.add_trace(
                go.Scatter(x=_locs, y=_hgts, mode="markers", name="QFlex modes",
                            marker=dict(color="#2E8B57", size=10, symbol="diamond",
                                        line=dict(color="white", width=1))),
                row=1, col=2,
            )

    _fig.update_xaxes(title_text="Cumulative probability", row=1, col=1)
    _fig.update_yaxes(title_text="Value", row=1, col=1)
    _fig.update_xaxes(title_text="Value", row=1, col=2)
    _fig.update_yaxes(title_text="Density", row=1, col=2)
    _fig.update_layout(height=430, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.18))

    def _mode_line(name, fit, modes):
        if fit is None:
            return f"**{name}:** fit failed"
        _n_modes = modes[0]
        _feas = "valid" if fit.is_feasible else "⚠️ infeasible (PDF goes negative)"
        _shape = "unimodal" if _n_modes == 1 else (f"**{_n_modes} modes — spurious structure**" if _n_modes and _n_modes > 1 else "no modes detected")
        return f"**{name}:** {_feas}, {_shape}"

    _summary = mo.md(
        f"{_mode_line('Metalog', metalog_fit, metalog_modes)}  \n"
        f"{_mode_line('QFlex', qflex_fit, qflex_modes)}"
        + (f"\n\n⚠️ {fit_error}" if fit_error else "")
    )

    mo.vstack([mo.ui.plotly(_fig), _summary])
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Empirical case studies

        The paper's three real datasets — each panel shows the empirical
        quantile function (the raw sorted data as a step-like curve) with a
        pointwise 95% bootstrap confidence band. **Drag a rectangle on the
        plot to zoom** into any region (the tails are usually where the band
        widens the most); double-click to reset.
        """
    )
    return


@app.cell
def _(DATA_DIR, np, os, pd):
    def _load_fish():
        _df = pd.read_excel(str(DATA_DIR / "Fish Biology.xlsx"))
        return np.sort(_df["Fish Weight (lbs)"].dropna().values.astype(float))

    def _load_hydrology():
        _df = pd.read_excel(str(DATA_DIR / "Hydrology.xlsx"))
        return np.sort(_df["Gauge Height (ft)"].dropna().values.astype(float))

    def _load_geyser():
        _df = pd.read_csv(
            str(DATA_DIR / "geyser.txt"), sep=r"\s+", header=None,
            names=["eruption", "waiting", "indicator"],
        )
        return np.sort(_df["waiting"].dropna().values.astype(float))

    EMPIRICAL_DATASETS = {
        "Fish weights (Fish Biology.xlsx)": (_load_fish, "Weight (lbs)"),
        "River gauge height (Hydrology.xlsx)": (_load_hydrology, "Gauge height (ft)"),
        "Old Faithful waiting time (geyser.txt)": (_load_geyser, "Waiting time (min)"),
    }
    return (EMPIRICAL_DATASETS,)


@app.cell
def _(EMPIRICAL_DATASETS, mo):
    empirical_dataset = mo.ui.dropdown(
        options=list(EMPIRICAL_DATASETS.keys()),
        value="Fish weights (Fish Biology.xlsx)",
        label="Dataset",
    )
    n_boot_eqf = mo.ui.slider(
        start=100, stop=1000, step=100, value=300,
        label="Bootstrap resamples for the CI band", show_value=True,
    )
    mo.hstack([empirical_dataset, n_boot_eqf], justify="start", gap=2)
    return empirical_dataset, n_boot_eqf


@app.cell
def _(EMPIRICAL_DATASETS, empirical_dataset, n_boot_eqf, np):
    # Pointwise 95% percentile bootstrap CI on the EQF, matching the method
    # used for the paper's own EQF+CI figures (e.g. Figure 10, hydrology):
    # resample the raw data with replacement, interpolate each resample's EQF
    # onto a common probability grid, and take the 2.5th/97.5th percentiles
    # at each grid point across resamples.
    _loader, eqf_axis_label = EMPIRICAL_DATASETS[empirical_dataset.value]
    eqf_x_raw = _loader()
    _N = len(eqf_x_raw)
    _p_emp = np.arange(1, _N + 1) / (_N + 1)
    eqf_p_grid = np.linspace(0.01, 0.99, 300)

    _rng = np.random.default_rng(42)
    _boot = np.empty((n_boot_eqf.value, len(eqf_p_grid)))
    for _b in range(n_boot_eqf.value):
        _x_boot = np.sort(_rng.choice(eqf_x_raw, size=_N, replace=True))
        _boot[_b] = np.interp(eqf_p_grid, _p_emp, _x_boot)

    eqf_point = np.interp(eqf_p_grid, _p_emp, eqf_x_raw)
    eqf_lo = np.percentile(_boot, 2.5, axis=0)
    eqf_hi = np.percentile(_boot, 97.5, axis=0)
    return eqf_axis_label, eqf_hi, eqf_lo, eqf_p_grid, eqf_point, eqf_x_raw


@app.cell
def _(
    eqf_axis_label,
    eqf_hi,
    eqf_lo,
    eqf_p_grid,
    eqf_point,
    eqf_x_raw,
    empirical_dataset,
    go,
    mo,
    np,
):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=np.concatenate([eqf_p_grid, eqf_p_grid[::-1]]),
        y=np.concatenate([eqf_hi, eqf_lo[::-1]]),
        fill="toself", fillcolor="rgba(59, 95, 160, 0.18)",
        line=dict(width=0), hoverinfo="skip", name="95% bootstrap CI",
    ))
    _fig.add_trace(go.Scatter(
        x=eqf_p_grid, y=eqf_point, mode="lines", name="Empirical QF",
        line=dict(color="#3B5FA0", width=2.4),
    ))
    _n_raw = len(eqf_x_raw)
    _p_raw = (np.arange(1, _n_raw + 1) - 0.3) / (_n_raw + 0.4)
    _stride = max(1, _n_raw // 400)  # thin the raw-point overlay for large N
    _fig.add_trace(go.Scatter(
        x=_p_raw[::_stride], y=eqf_x_raw[::_stride], mode="markers", name="Raw data",
        marker=dict(color="#4E5972", size=4, opacity=0.35),
    ))
    _fig.update_layout(
        title=f"{empirical_dataset.value} — N={_n_raw}",
        xaxis_title="Cumulative probability", yaxis_title=eqf_axis_label,
        height=420, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.18),
        dragmode="zoom",
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Full-simulation mode

        Run many replicates of the scenario configured above (same family,
        sampling mode, N, and K's) and watch the table fill in live &mdash;
        feasibility rate, false-modality rate, and median W1 distance to the
        true quantile function, the same statistics behind the paper's Table
        2/3. Capped at 100 replicates to keep this responsive in-browser.
        """
    )
    return


@app.cell
def _(mo):
    n_replicates = mo.ui.slider(
        start=5, stop=100, step=5, value=30, label="Replicates", show_value=True
    )
    run_batch = mo.ui.run_button(label="▶ Run batch")
    mo.hstack([n_replicates, run_batch], justify="start", gap=2)
    return n_replicates, run_batch


@app.cell
def _(
    ConstraintType,
    Metalog,
    MetalogError,
    QFlex,
    QFlexError,
    base_seed,
    detect_modes_from_arrays,
    k_metalog,
    k_qflex,
    mo,
    n_effective,
    n_replicates,
    np,
    qflex_constraint,
    reference_sample,
    run_batch,
    sampling_mode,
    true_dist,
):
    # Streamed batch run: each replicate re-draws a sample (Monte Carlo or
    # bootstrap, per the sampling-mode control above) and fits both models,
    # replacing the cell's output after every replicate so the table fills in
    # live instead of blocking until the whole batch finishes.
    _rows = []

    if run_batch.value:
        _ygrid = np.linspace(0.01, 0.99, 1000)
        _dp = _ygrid[1] - _ygrid[0]
        _rng = np.random.default_rng(base_seed.value + 777)
        _constraint = ConstraintType[qflex_constraint.value]
        _x_true = true_dist.quantile(_ygrid)
        _n_reps = n_replicates.value

        for _rep in range(_n_reps):
            if sampling_mode.value.startswith("Monte Carlo"):
                _x = np.sort(true_dist.quantile(_rng.random(n_effective)))
            else:
                _x = np.sort(_rng.choice(reference_sample, size=n_effective, replace=True))
            _n = len(_x)
            _y = (np.arange(1, _n + 1) - 0.3) / (_n + 0.4)

            for _model_name, _fit_fn in (
                ("Metalog", lambda: Metalog(_x, _y, terms=k_metalog.value)),
                ("QFlex", lambda: QFlex(_x, _y, terms=k_qflex.value, constraint_type=_constraint)),
            ):
                try:
                    _fit = _fit_fn()
                    _xg = _fit.quantile(_ygrid)
                    _pg = _fit.pdf(_ygrid)
                    _n_modes, _, _ = detect_modes_from_arrays(_xg, _pg)
                    _w1 = float(np.sum(np.abs(_xg - _x_true)) * _dp)
                    _rows.append({
                        "Replicate": _rep + 1,
                        "Model": _model_name,
                        "Feasible": bool(_fit.is_feasible),
                        "Modes": int(_n_modes) if _n_modes is not None else 0,
                        "W1 vs true QF": round(_w1, 4),
                    })
                except (MetalogError, QFlexError):
                    _rows.append({
                        "Replicate": _rep + 1,
                        "Model": _model_name,
                        "Feasible": False,
                        "Modes": None,
                        "W1 vs true QF": None,
                    })

            mo.output.replace(
                mo.vstack([
                    mo.md(f"Running replicate {_rep + 1} / {_n_reps}..."),
                    mo.ui.table(_rows, selection=None, show_download=False, pagination=True, page_size=10),
                ])
            )

        mo.output.replace(
            mo.vstack([
                mo.md(f"**Done — {_n_reps} replicates.**"),
                mo.ui.table(_rows, selection=None, show_download=False, pagination=True, page_size=10),
            ])
        )
    else:
        mo.output.replace(
            mo.md("*Click **▶ Run batch** above to simulate replicates with the current controls.*")
        )

    batch_rows = _rows
    return (batch_rows,)


@app.cell
def _(batch_rows, mo, pd):
    if batch_rows:
        _df = pd.DataFrame(batch_rows)
        _summary_rows = []
        for _model, _g in _df.groupby("Model", sort=False):
            _n = len(_g)
            _feas = _g[_g["Feasible"]]
            _feas_pct = round(100 * len(_feas) / _n) if _n else 0
            _false_modal_pct = round(100 * (_feas["Modes"] > 1).mean()) if len(_feas) else 0
            _median_w1 = round(_feas["W1 vs true QF"].median(), 4) if len(_feas) else float("nan")
            _summary_rows.append({
                "Model": _model,
                "Replicates": _n,
                "Feasibility %": _feas_pct,
                "False-modality %": _false_modal_pct,
                "Median W1 vs true QF": _median_w1,
            })
        _out = mo.vstack([
            mo.md("**Summary across all replicates** &mdash; the same statistics behind the paper's Table 2/3."),
            mo.ui.table(_summary_rows, selection=None, show_download=False, pagination=False),
        ])
    else:
        _out = mo.md("*Summary stats appear here once a batch has run.*")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        *Built as a companion to Khanna &amp; Bickel, "Inferring Distributional
        Features based on Quantile-Parameterized Distribution Fits." All four
        playground panels above are live.*
        """
    )
    return


if __name__ == "__main__":
    app.run()
