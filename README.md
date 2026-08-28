# QPD Playground

A live, in-browser interactive companion to Khanna & Bickel, *Inferring
Distributional Features based on Quantile-Parameterized Distribution
Fits*, built as a [marimo](https://marimo.io) notebook.

**Live notebook:** https://rohit5khanna.github.io/QPD-Distributional-Features/

The page runs entirely in your browser via WebAssembly (Pyodide) — no
server, no install. Redraw samples, refit, and explore, and the figures
recompute live.

## What's inside

- **Live redraw & refit** — draw a fresh sample from a chosen Johnson
  distribution, fit Metalog and QFlex, and watch spurious modes appear.
- **Sample-size playground** — a free-form N slider with a K-aware floor.
- **Empirical case studies** — EQF + 95% bootstrap CI for the paper's
  three real datasets (fish weights, river gauge height, Old Faithful).
- **Full-simulation batch mode** — run up to 100 replicates of the
  configured scenario and watch a summary-stats table fill in live.

## Run / edit locally

```bash
pip install -r requirements.txt
marimo edit app.py      # interactive editor
# or preview the exact WASM build that gets deployed:
marimo export html-wasm app.py -o _site --mode run
python -m http.server --directory _site   # then open the printed URL
```

## How the browser build works

`common/` (the `metalog`, `qflex`, and `jpse` fitting/reference code) is
a local package, not a PyPI package — marimo automatically bundles it as
a wheel at export time, so no manual build step is needed for it. The
notebook's dependency header at the top of `app.py` explicitly lists the
third-party packages (`numpy`, `pandas`, `plotly`, `scipy`, `openpyxl`)
since marimo's WASM export only auto-resolves what it can detect from
imports, and a couple of these (`plotly` in particular) aren't part of
Pyodide's default bundle. `public/` holds the three empirical datasets;
marimo copies this folder into the deployed site automatically.
