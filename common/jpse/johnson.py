"""
Johnson Distribution System - Quantile Function Implementation

This module implements the three Johnson distribution families using their
exact quantile function forms as specified in Bickel (2026).

The Johnson system is defined by:
    Z = γ + δ*g((X - ξ)/λ), where Z ~ N(0,1)

Solving for X gives the quantile functions:
    Q(p) = ξ + λ*g^(-1)((Φ^(-1)(p) - γ)/δ)

NOTATION. ξ location, λ scale, γ and δ shape -- the classical Johnson symbols,
as used both by the manuscript and by Bickel (2026), so formulas quoted from
either carry over unchanged. In code λ is spelled `lam`, because `lambda` is a
Python keyword.

Until this was corrected, the module used η, κ, c, d instead. That was this
module's own departure from the notation of the paper it implements -- NOT
Bickel's notation, despite sitting directly under the reference below.

where Φ^(-1)(p) is the standard normal quantile function.

Reference:
    Bickel, J. E. (2026). Quantile-based power-series expansions of the Johnson
    distribution system. Communications in Statistics - Theory and Methods.
    DOI: 10.1080/03610926.2025.2612230
"""

import numpy as np
from scipy.stats import norm
from scipy.special import expit  # logistic sigmoid = logit^(-1)


class JohnsonBase:
    """
    Base class for Johnson distributions.

    Parameters
    ----------
    xi : float
        Location parameter (ξ)
    lam : float
        Scale parameter (λ > 0)
    gamma : float
        Shape parameter
    delta : float
        Shape/scale parameter (delta > 0)
    """

    def __init__(self, xi=0.0, lam=1.0, gamma=0.5, delta=1.2):
        if lam <= 0:
            raise ValueError("lam must be positive")
        if delta <= 0:
            raise ValueError("delta must be positive")

        self.xi = xi
        self.lam = lam
        self.gamma = gamma
        self.delta = delta

    def _standard_normal_quantile(self, p):
        """Compute Φ^(-1)(p) - the standard normal quantile function"""
        return norm.ppf(p)

    def _centered_scaled_normal(self, p):
        """Compute (Φ^(-1)(p) - gamma) / delta"""
        return (self._standard_normal_quantile(p) - self.gamma) / self.delta

    def quantile(self, p):
        """
        Quantile function Q(p).

        Must be implemented by subclasses.

        Parameters
        ----------
        p : array_like
            Probability values in (0, 1)

        Returns
        -------
        x : ndarray
            Quantile values
        """
        raise NotImplementedError("Subclasses must implement quantile()")

    def pdf(self, x, grid_size=1000):
        """
        Probability density function using the quantile-density relation.

        The PDF is computed as f(Q(p)) = 1 / q(p), where q(p) = dQ(p)/dp
        is the quantile density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the PDF
        grid_size : int
            Number of grid points for computing the inverse mapping

        Returns
        -------
        pdf : ndarray
            PDF values at x
        """
        x = np.asarray(x)

        # Create fine grid of p values
        p_grid = np.linspace(0.001, 0.999, grid_size)
        x_grid = self.quantile(p_grid)
        pdf_grid = self.pdf_via_quantile_density(p_grid)

        # Interpolate to get PDF at requested x values
        pdf = np.interp(x, x_grid, pdf_grid, left=0.0, right=0.0)

        return pdf

    def pdf_via_quantile_density(self, p, step_size=1e-6):
        """
        Compute PDF at Q(p) using the quantile-density relation.

        f(Q(p)) = 1 / q(p), where q(p) = dQ(p)/dp

        Parameters
        ----------
        p : array_like
            Probability values in (0, 1)
        step_size : float
            Step size for numerical differentiation

        Returns
        -------
        pdf : ndarray
            PDF values at Q(p)
        """
        p = np.asarray(p)

        # Numerical derivative: q(p) ≈ (Q(p+h) - Q(p-h)) / (2h)
        h = step_size
        p_plus = np.clip(p + h, 1e-10, 1 - 1e-10)
        p_minus = np.clip(p - h, 1e-10, 1 - 1e-10)

        q_p = (self.quantile(p_plus) - self.quantile(p_minus)) / (2 * h)

        # Ensure positive quantile density
        q_p = np.clip(q_p, 1e-12, None)

        # f(Q(p)) = 1 / q(p)
        return 1.0 / q_p

    def rvs(self, size=1, random_state=None):
        """
        Generate random variates.

        Uses the probability integral transform: X = Q(U) where U ~ Uniform(0,1)

        Parameters
        ----------
        size : int or tuple
            Output shape
        random_state : int, RandomState, or None
            Random seed

        Returns
        -------
        samples : ndarray
            Random samples
        """
        if random_state is not None:
            np.random.seed(random_state)

        u = np.random.uniform(0, 1, size=size)
        return self.quantile(u)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"xi={self.xi}, lam={self.lam}, "
                f"gamma={self.gamma}, delta={self.delta})")


class JohnsonSU(JohnsonBase):
    """
    Johnson SU distribution (unbounded support: X ∈ (-∞, ∞)).

    Quantile function:
        Q_SU(p) = ξ + λ * sinh((Φ^(-1)(p) - gamma) / delta)

    where sinh is the hyperbolic sine function.

    Parameters
    ----------
    xi : float
        Location parameter (default 0.0)
    lam : float
        Scale parameter (default 1.0, must be > 0)
    gamma : float
        Shape parameter (default 0.5)
    delta : float
        Shape/scale parameter (default 1.2, must be > 0)

    Examples
    --------
    >>> # Figure 1 parameters from Bickel (2026)
    >>> dist = JohnsonSU(xi=0, lam=1, gamma=0.5, delta=1.2)
    >>> samples = dist.rvs(size=100000)
    >>> x = dist.quantile(0.5)  # Median
    """

    def quantile(self, p):
        """
        Quantile function for Johnson SU.

        Q_SU(p) = ξ + λ * sinh((Φ^(-1)(p) - gamma) / delta)

        Parameters
        ----------
        p : array_like
            Probability values in (0, 1)

        Returns
        -------
        x : ndarray
            Quantile values (unbounded)
        """
        p = np.asarray(p)
        z = self._centered_scaled_normal(p)
        return self.xi + self.lam * np.sinh(z)


class JohnsonSL(JohnsonBase):
    """
    Johnson SL distribution (semi-bounded support: X ∈ (ξ, ∞)).

    Quantile function:
        Q_SL(p) = ξ + λ * exp((Φ^(-1)(p) - gamma) / delta)

    where exp is the exponential function.

    Parameters
    ----------
    xi : float
        Location parameter / lower bound (default 0.0)
    lam : float
        Scale parameter (default 1.0, must be > 0)
    gamma : float
        Shape parameter (default 0.5)
    delta : float
        Shape/scale parameter (default 1.2, must be > 0)

    Examples
    --------
    >>> # Figure 1 parameters from Bickel (2026)
    >>> dist = JohnsonSL(xi=0, lam=1, gamma=0.5, delta=1.2)
    >>> samples = dist.rvs(size=100000)
    >>> x = dist.quantile(0.5)  # Median
    """

    def quantile(self, p):
        """
        Quantile function for Johnson SL.

        Q_SL(p) = ξ + λ * exp((Φ^(-1)(p) - gamma) / delta)

        Parameters
        ----------
        p : array_like
            Probability values in (0, 1)

        Returns
        -------
        x : ndarray
            Quantile values (> ξ)
        """
        p = np.asarray(p)
        z = self._centered_scaled_normal(p)
        return self.xi + self.lam * np.exp(z)


class JohnsonSB(JohnsonBase):
    """
    Johnson SB distribution (bounded support: X ∈ (ξ, ξ + λ)).

    Quantile function:
        Q_SB(p) = ξ + λ * logit^(-1)((Φ^(-1)(p) - gamma) / delta)

    where logit^(-1)(x) = 1/(1 + exp(-x)) is the logistic sigmoid function.

    Parameters
    ----------
    xi : float
        Lower bound (default 0.0)
    lam : float
        Width of support (default 1.0, must be > 0)
        Support is (ξ, ξ + λ)
    gamma : float
        Shape parameter (default 0.5)
    delta : float
        Shape/scale parameter (default 1.2, must be > 0)

    Examples
    --------
    >>> # Figure 1 parameters from Bickel (2026)
    >>> dist = JohnsonSB(xi=0, lam=1, gamma=0.5, delta=1.2)
    >>> samples = dist.rvs(size=100000)
    >>> x = dist.quantile(0.5)  # Median
    >>> # All samples will be in (0, 1)
    """

    def quantile(self, p):
        """
        Quantile function for Johnson SB.

        Q_SB(p) = ξ + λ * logit^(-1)((Φ^(-1)(p) - gamma) / delta)

        where logit^(-1)(x) = 1/(1 + exp(-x))

        Parameters
        ----------
        p : array_like
            Probability values in (0, 1)

        Returns
        -------
        x : ndarray
            Quantile values in (ξ, ξ + λ)
        """
        p = np.asarray(p)
        z = self._centered_scaled_normal(p)
        # expit(x) = 1/(1 + exp(-x)) is the logistic sigmoid
        return self.xi + self.lam * expit(z)
