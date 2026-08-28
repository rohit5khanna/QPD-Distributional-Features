"""
Shared mode-detection utilities used across experiments.

This file intentionally matches the mode-detection logic used in:
`Modality Tests/MC_UnimodalTruth/mc_unimodal_experiment.py::count_modes_in_pdf`

Key properties:
- Evaluate PDF on probability grid y ∈ [0.01, 0.99]
- Use scipy.signal.find_peaks with prominence = 0.01 * max(PDF)
- Return -1 for invalid PDF values or if mode detection errors

Updated: 2026-02-04
- Prominence threshold is set to 1% to preserve historical experiment behavior
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_modes_from_arrays(x_vals, pdf_vals):
    """
    Detect mode count, locations, and heights from precomputed x/pdf arrays.

    This avoids re-evaluating model.quantile() and model.pdf() when those
    arrays were already computed upstream.
    """
    try:
        x_vals = np.asarray(x_vals)
        pdf_vals = np.asarray(pdf_vals)

        if x_vals.shape != pdf_vals.shape:
            return -1, None, None

        if not np.all(np.isfinite(pdf_vals)) or np.max(pdf_vals) <= 0:
            return -1, None, None

        prominence_threshold = 0.01 * pdf_vals.max()
        peaks, _properties = find_peaks(pdf_vals, prominence=prominence_threshold)

        if len(peaks) == 0:
            return 0, np.array([]), np.array([])

        mode_locations = x_vals[peaks]
        mode_heights = pdf_vals[peaks]
        return int(len(peaks)), mode_locations, mode_heights
    except Exception:
        return -1, None, None


def count_modes_in_pdf(model, n_points: int = 1000) -> int:
    """
    Count modes by detecting local maxima in the PDF.

    Parameters
    ----------
    model : object
        Fitted distribution model with quantile() and pdf() methods.
    n_points : int
        Number of points to evaluate PDF at.

    Returns
    -------
    n_modes : int
        Number of modes detected.
        Returns -1 for invalid PDF or on error (matches MC_UnimodalTruth).
    """
    try:
        y_grid = np.linspace(0.01, 0.99, n_points)
        _x_vals = model.quantile(y_grid)
        pdf_vals = model.pdf(y_grid)

        # Check for invalid PDF values
        if not np.all(np.isfinite(pdf_vals)) or np.max(pdf_vals) <= 0:
            return -1  # Invalid PDF

        n_modes, _locs, _hgts = detect_modes_from_arrays(_x_vals, pdf_vals)
        return int(n_modes)
    except Exception:
        return -1  # Error in mode detection


def detect_modes_in_pdf(model, n_points: int = 1000):
    """
    Detect mode locations and heights using the same logic as count_modes_in_pdf().

    Returns
    -------
    n_modes : int
        Number of modes, or -1 if invalid PDF / error.
    mode_locations : np.ndarray | None
        Locations (x) of detected modes. None if n_modes == -1.
    mode_heights : np.ndarray | None
        PDF values at detected modes. None if n_modes == -1.
    """
    try:
        y_grid = np.linspace(0.01, 0.99, n_points)
        x_vals = model.quantile(y_grid)
        pdf_vals = model.pdf(y_grid)

        return detect_modes_from_arrays(x_vals, pdf_vals)
    except Exception:
        return -1, None, None
