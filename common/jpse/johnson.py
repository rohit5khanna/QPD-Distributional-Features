"""
Johnson Distribution System - Quantile Function Implementation

This module implements the three Johnson distribution families using their
exact quantile function forms as specified in Bickel (2026).

The Johnson system is defined by:
    Z = c + d*g((X - η)/κ), where Z ~ N(0,1)

Solving for X gives the quantile functions:
    Q(p) = η + κ*g^(-1)((Φ^(-1)(p) - c)/d)

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
    eta : float
        Location parameter (η)
    kappa : float
        Scale parameter (κ > 0)
    c : float
        Shape parameter
    d : float
        Shape/scale parameter (d > 0)
    """

    def __init__(self, eta=0.0, kappa=1.0, c=0.5, d=1.2):
        if kappa <= 0:
            raise ValueError("kappa must be positive")
        if d <= 0:
            raise ValueError("d must be positive")

        self.eta = eta
        self.kappa = kappa
        self.c = c
        self.d = d

    def _standard_normal_quantile(self, p):
        """Compute Φ^(-1)(p) - the standard normal quantile function"""
        return norm.ppf(p)

    def _centered_scaled_normal(self, p):
        """Compute (Φ^(-1)(p) - c) / d"""
        return (self._standard_normal_quantile(p) - self.c) / self.d

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
                f"eta={self.eta}, kappa={self.kappa}, "
                f"c={self.c}, d={self.d})")


class JohnsonSU(JohnsonBase):
    """
    Johnson SU distribution (unbounded support: X ∈ (-∞, ∞)).

    Quantile function:
        Q_SU(p) = η + κ * sinh((Φ^(-1)(p) - c) / d)

    where sinh is the hyperbolic sine function.

    Parameters
    ----------
    eta : float
        Location parameter (default 0.0)
    kappa : float
        Scale parameter (default 1.0, must be > 0)
    c : float
        Shape parameter (default 0.5)
    d : float
        Shape/scale parameter (default 1.2, must be > 0)

    Examples
    --------
    >>> # Figure 1 parameters from Bickel (2026)
    >>> dist = JohnsonSU(eta=0, kappa=1, c=0.5, d=1.2)
    >>> samples = dist.rvs(size=100000)
    >>> x = dist.quantile(0.5)  # Median
    """

    def quantile(self, p):
        """
        Quantile function for Johnson SU.

        Q_SU(p) = η + κ * sinh((Φ^(-1)(p) - c) / d)

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
        return self.eta + self.kappa * np.sinh(z)


class JohnsonSL(JohnsonBase):
    """
    Johnson SL distribution (semi-bounded support: X ∈ (η, ∞)).

    Quantile function:
        Q_SL(p) = η + κ * exp((Φ^(-1)(p) - c) / d)

    where exp is the exponential function.

    Parameters
    ----------
    eta : float
        Location parameter / lower bound (default 0.0)
    kappa : float
        Scale parameter (default 1.0, must be > 0)
    c : float
        Shape parameter (default 0.5)
    d : float
        Shape/scale parameter (default 1.2, must be > 0)

    Examples
    --------
    >>> # Figure 1 parameters from Bickel (2026)
    >>> dist = JohnsonSL(eta=0, kappa=1, c=0.5, d=1.2)
    >>> samples = dist.rvs(size=100000)
    >>> x = dist.quantile(0.5)  # Median
    """

    def quantile(self, p):
        """
        Quantile function for Johnson SL.

        Q_SL(p) = η + κ * exp((Φ^(-1)(p) - c) / d)

        Parameters
        ----------
        p : array_like
            Probability values in (0, 1)

        Returns
        -------
        x : ndarray
            Quantile values (> η)
        """
        p = np.asarray(p)
        z = self._centered_scaled_normal(p)
        return self.eta + self.kappa * np.exp(z)


class JohnsonSB(JohnsonBase):
    """
    Johnson SB distribution (bounded support: X ∈ (η, η + κ)).

    Quantile function:
        Q_SB(p) = η + κ * logit^(-1)((Φ^(-1)(p) - c) / d)

    where logit^(-1)(x) = 1/(1 + exp(-x)) is the logistic sigmoid function.

    Parameters
    ----------
    eta : float
        Lower bound (default 0.0)
    kappa : float
        Width of support (default 1.0, must be > 0)
        Support is (η, η + κ)
    c : float
        Shape parameter (default 0.5)
    d : float
        Shape/scale parameter (default 1.2, must be > 0)

    Examples
    --------
    >>> # Figure 1 parameters from Bickel (2026)
    >>> dist = JohnsonSB(eta=0, kappa=1, c=0.5, d=1.2)
    >>> samples = dist.rvs(size=100000)
    >>> x = dist.quantile(0.5)  # Median
    >>> # All samples will be in (0, 1)
    """

    def quantile(self, p):
        """
        Quantile function for Johnson SB.

        Q_SB(p) = η + κ * logit^(-1)((Φ^(-1)(p) - c) / d)

        where logit^(-1)(x) = 1/(1 + exp(-x))

        Parameters
        ----------
        p : array_like
            Probability values in (0, 1)

        Returns
        -------
        x : ndarray
            Quantile values in (η, η + κ)
        """
        p = np.asarray(p)
        z = self._centered_scaled_normal(p)
        # expit(x) = 1/(1 + exp(-x)) is the logistic sigmoid
        return self.eta + self.kappa * expit(z)
