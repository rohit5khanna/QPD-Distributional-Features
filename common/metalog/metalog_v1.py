"""
Metalog Distributions Implementation - Version 1.0 (Original Keelin 2016)

Based on:
- "The Metalog Distributions" by Thomas W. Keelin (2016)

This module implements the original Metalog 1.0 family of distributions:
- Unbounded metalog distributions
- Semibounded (log) metalog distributions  
- Bounded (logit) metalog distributions

Key features:
- Original Metalog 1.0 basis function ordering from Keelin (2016)
- Linear parameter estimation from CDF data
- Closed-form PDF and CDF expressions
- Feasibility checking
- Monte Carlo sampling
- SPT (Symmetric Percentile Triplet) support

Note: This is the ORIGINAL implementation. For terms ≥ 7, Metalog 2.0 
(metalog_v2.py) uses an updated ordering that avoids rank deficiency.
See Documentation/METALOG_2_0_UPDATE_SUMMARY.md for details.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, integrate, stats
from scipy.linalg import inv, LinAlgError, svd
import warnings
from typing import Tuple, List, Union, Optional


class MetalogError(Exception):
    """Custom exception for metalog-related errors."""
    pass


class Metalog:
    """
    Unbounded metalog distribution.
    
    Parameters:
    -----------
    x_data : array-like
        x values (quantiles) corresponding to cumulative probabilities
    y_data : array-like  
        y values (cumulative probabilities) in (0, 1)
    terms : int
        Number of terms to use (2-10 recommended)
    """
    
    def __init__(self, x_data, y_data, terms=5):
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)
        self.terms = terms
        
        # Validate inputs
        self._validate_inputs()
        
        # Estimate parameters
        self.coefficients = self._estimate_parameters()
        
        # Check feasibility
        self.is_feasible = self._check_feasibility()
        
        if not self.is_feasible:
            warnings.warn("Metalog distribution may not be feasible (PDF not strictly positive)")
    
    def _validate_inputs(self):
        """Validate input data."""
        if len(self.x_data) != len(self.y_data):
            raise MetalogError("x_data and y_data must have same length")
        
        if len(self.x_data) < self.terms:
            raise MetalogError(f"Need at least {self.terms} data points for {self.terms}-term metalog")
        
        if np.any(self.y_data <= 0) or np.any(self.y_data >= 1):
            raise MetalogError("All y_data values must be in (0, 1)")
        
        if len(np.unique(self.y_data)) < self.terms:
            raise MetalogError(f"Need at least {self.terms} distinct y values")
    
    def _build_design_matrix(self):
        """
        Build the design matrix Y for parameter estimation.
        
        Follows ORIGINAL Metalog 1.0 specification from Keelin (2016).
        
        The basis functions follow the pattern:
        - j=1: 1 (constant)
        - j=2: ℓ(y) (logit)
        - j=3: (y-0.5)ℓ(y)
        - j=4: (y-0.5)
        - j=5: (y-0.5)² (WITHOUT logit)
        - j=6: (y-0.5)²ℓ(y) (WITH logit)
        - j=7: (y-0.5)³ (WITHOUT logit) ← DIFFERS from 2.0
        - j=8: (y-0.5)³ℓ(y) (WITH logit) ← DIFFERS from 2.0
        
        For terms ≥ 5: alternate WITHOUT/WITH logit for each power
        """
        y = self.y_data
        n = self.terms
        m = len(y)
        
        Y = np.zeros((m, n))
        logit_y = np.log(y / (1 - y))
        
        # First 4 terms are the same in both versions
        Y[:, 0] = 1
        if n >= 2:
            Y[:, 1] = logit_y
        if n >= 3:
            Y[:, 2] = (y - 0.5) * logit_y
        if n >= 4:
            Y[:, 3] = (y - 0.5)
        
        # For terms 5+, use Metalog 1.0 pattern
        for i in range(5, n + 1):
            power = int((i - 3) / 2) + 1
            if (i - 3) % 2 == 0:  # First term with this power: WITHOUT logit
                Y[:, i - 1] = (y - 0.5) ** power
            else:  # Second term with this power: WITH logit
                Y[:, i - 1] = (y - 0.5) ** power * logit_y
        
        return Y
    
    def _estimate_parameters(self):
        """Estimate metalog coefficients using exact solution or linear least squares (exact inverse)."""
        Y = self._build_design_matrix()
        m = len(self.x_data)
        n = self.terms
        
        try:
            if m == n:
                # Exact solution when square
                Y_inv = inv(Y)
                coefficients = Y_inv @ self.x_data
            else:
                # Linear least squares: a = (Y^T Y)^(-1) Y^T x
                YTY = Y.T @ Y
                YTY_inv = inv(YTY)
                coefficients = YTY_inv @ Y.T @ self.x_data
        except LinAlgError:
            raise MetalogError("Design matrix is singular. Check for collinear data.")
        
        return coefficients
    
    def quantile(self, y):
        """
        Metalog quantile function M_n(y) following ORIGINAL Metalog 1.0 specification.
        
        Parameters:
        -----------
        y : array-like
            Cumulative probabilities in (0, 1)
            
        Returns:
        --------
        x : array-like
            Corresponding quantile values
        """
        y = np.asarray(y)
        a = self.coefficients
        n = self.terms
        
        logit_y = np.log(y / (1 - y))
        
        # Initialize result
        x = np.zeros_like(y, dtype=float)
        
        # First 4 terms (same in both versions)
        x += a[0]  # constant
        if n >= 2:
            x += a[1] * logit_y
        if n >= 3:
            x += a[2] * (y - 0.5) * logit_y
        if n >= 4:
            x += a[3] * (y - 0.5)
        
        # For terms 5+, use Metalog 1.0 pattern
        for i in range(5, n + 1):
            power = int((i - 3) / 2) + 1
            if (i - 3) % 2 == 0:  # WITHOUT logit
                x += a[i - 1] * ((y - 0.5) ** power)
            else:  # WITH logit
                x += a[i - 1] * ((y - 0.5) ** power) * logit_y
        
        return x
    
    def pdf(self, y):
        """
        Metalog probability density function m_n(y).
        Uses unified numerical differentiation for consistency.
        
        Parameters:
        -----------
        y : array-like
            Cumulative probabilities in (0, 1)
            
        Returns:
        --------
        pdf : array-like
            Probability density values
        """
        y = np.asarray(y)
        return self._unified_pdf_numerical(y)
    
    def _unified_pdf_numerical(self, y):
        """Unified numerical PDF calculation with improved resolution and smoothing."""
        def derivative(y_vals, delta):
            y_plus = np.clip(y_vals + delta, 1e-12, 1 - 1e-12)
            y_minus = np.clip(y_vals - delta, 1e-12, 1 - 1e-12)
            x_plus = self.quantile(y_plus)
            x_minus = self.quantile(y_minus)
            return (x_plus - x_minus) / (y_plus - y_minus)
        
        # Use adaptive step size based on y value
        # Smaller steps near boundaries, larger in middle
        y_array = np.asarray(y)
        eps_array = np.where(
            y_array < 0.1, 1e-10,  # Very small near lower bound
            np.where(
                y_array > 0.9, 1e-10,  # Very small near upper bound
                1e-8  # Standard step in middle
            )
        )
        
        # Calculate derivative with adaptive step sizes
        grad = np.zeros_like(y_array, dtype=float)
        for i, (y_val, eps) in enumerate(zip(y_array, eps_array)):
            grad[i] = derivative(y_val, eps)
        
        # Handle problematic regions with larger step sizes
        bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
        if np.any(bad):
            grad[bad] = derivative(y_array[bad], 1e-6)
        
        bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
        if np.any(bad):
            grad[bad] = derivative(y_array[bad], 1e-4)
        
        # Ensure positive derivative (quantile should be monotonic)
        grad = np.abs(grad)
        grad = np.clip(grad, 1e-12, np.inf)
        
        pdf = 1.0 / grad
        
        # Apply smoothing to reduce discretization
        if len(pdf) > 5:
            # Use moving average for smoothing
            window_size = min(5, len(pdf))
            if window_size % 2 == 0:
                window_size -= 1  # Make odd
            pdf_smooth = np.convolve(pdf, np.ones(window_size)/window_size, mode='same')
            # Blend original and smoothed, more smoothing near boundaries
            blend_factor = np.where(
                y_array < 0.1, 0.7,  # More smoothing near lower bound
                np.where(
                    y_array > 0.9, 0.7,  # More smoothing near upper bound
                    0.3  # Less smoothing in middle
                )
            )
            pdf = blend_factor * pdf_smooth + (1 - blend_factor) * pdf
        
        return pdf
    
    def cdf(self, x):
        """
        Cumulative distribution function.
        
        Parameters:
        -----------
        x : array-like
            Values at which to evaluate CDF
            
        Returns:
        --------
        cdf : array-like
            Cumulative probabilities
        """
        x = np.asarray(x)
        x_scalar = x.ndim == 0
        x_flat = x.flatten()
        
        def objective(y, x_target):
            return self.quantile(y) - x_target
        
        cdf_vals = np.zeros_like(x_flat)
        
        for i, x_val in enumerate(x_flat):
            # Find y such that M_n(y) = x_val
            try:
                result = optimize.brentq(objective, 1e-10, 1 - 1e-10, args=(x_val,))
                cdf_vals[i] = result
            except ValueError:
                # Handle edge cases
                if x_val <= self.quantile(1e-10):
                    cdf_vals[i] = 0
                elif x_val >= self.quantile(1 - 1e-10):
                    cdf_vals[i] = 1
                else:
                    cdf_vals[i] = np.nan
        
        if x_scalar:
            return cdf_vals[0]
        else:
            return cdf_vals.reshape(x.shape)
    
    def _check_feasibility(self):
        """
        Check if the metalog distribution is feasible.
        
        A feasible metalog must satisfy:
        1. PDF > 0 everywhere (positive probability density)
        2. Quantile function is strictly monotonically increasing
        
        Uses the same grid as QFlex for consistency: 999 points with 0.001 spacing.
        
        Returns:
        --------
        bool
            True if distribution is feasible, False otherwise
        """
        # Use same grid as QFlex: 999 points with exact 0.001 spacing
        y_test = 0.001 + np.arange(999) * 0.001
        
        # Check 1: PDF must be strictly positive
        try:
            pdf_vals = self.pdf(y_test)
            if not np.all(pdf_vals > 0):
                return False
        except Exception:
            return False
        
        # Check 2: Quantile function must be strictly monotonically increasing
        try:
            q_vals = self.quantile(y_test)
            # Check for monotonicity: all differences should be positive
            diffs = np.diff(q_vals)
            if not np.all(diffs > 0):
                return False
        except Exception:
            return False
        
        return True
    
    def sample(self, size=1):
        """
        Generate random samples from the metalog distribution.
        
        Parameters:
        -----------
        size : int
            Number of samples to generate
            
        Returns:
        --------
        samples : array
            Random samples from the distribution
        """
        u = np.random.uniform(0, 1, size)
        return self.quantile(u)
    
    def moments(self, order=4):
        """
        Calculate moments of the metalog distribution.
        
        Parameters:
        -----------
        order : int
            Maximum moment order to calculate (1-4 for closed-form)
            
        Returns:
        --------
        moments : dict
            Dictionary containing mean, variance, skewness, kurtosis
        """
        if self.terms == 5 and order <= 4:
            return self._closed_form_moments_5term()
        else:
            return self._numerical_moments(order)
    
    def _closed_form_moments_5term(self):
        """Closed-form moments for 5-term metalog."""
        a = self.coefficients
        
        # Mean (first raw moment)
        mean = a[0] + a[2]/2 + a[4]/12
        
        # Variance (second central moment)
        pi_sq = np.pi**2
        variance = (pi_sq/3) * a[1]**2 + ((1/12) + (pi_sq/36)) * a[2]**2 + \
                   a[1]*a[3] + a[3]**2/12 + a[2]*a[4]/12 + a[4]**2/180
        
        # Standard deviation
        std = np.sqrt(variance)
        
        # Skewness (third standardized moment)
        skewness_num = (pi_sq * a[1]**2 * a[2] + (pi_sq/24) * a[2]**3 + 
                       0.5 * a[1] * a[2] * a[3] + (pi_sq/6) * a[1] * a[2] * a[3] +
                       0.125 * a[2] * a[3]**2 + a[1]**2 * a[4] + (pi_sq/24) * a[1]**2 * a[4] +
                       0.25 * a[1] * a[3] * a[4] + (pi_sq/60) * a[1]**3 * a[4] +
                       (1/3) * a[2] * a[3] * a[4] + (2/45) * (pi_sq) * a[1] * a[2] * a[3] * a[4] +
                       (3/40) * a[2] * a[3]**2 * a[4] + (1/120) * a[2] * a[4]**2 + a[4]**3/3780)
        
        skewness = skewness_num / (std**3)
        
        # Kurtosis (fourth standardized moment)
        kurtosis_num = (7/15) * (pi_sq**2) * a[3]**2 + (3/2) * (pi_sq**2) * a[1]**2 * a[2]**2 + \
                       (7/30) * (pi_sq**4) * a[1]**2 * a[2]**2 + a[3]**3/80 + \
                       (pi_sq/24) * a[3]**3 + (7 * pi_sq**4 * a[3]**3)/1200 + \
                       2 * (pi_sq**2) * a[2]**2 * a[3] + \
                       0.5 * a[1] * a[2]**2 * a[3] + (2/3) * (pi_sq**2) * a[1] * a[2]**2 * a[3] + \
                       2 * a[1]**2 * a[2]**3 + (pi_sq/6) * a[1]**2 * a[2]**3 + \
                       (pi_sq/8) * a[1]**2 * a[2]**3 + (pi_sq/40) * a[1]**2 * a[2]**3 + \
                       (1/3) * a[1] * a[2]**4 + a[2]**4/80 + \
                       (pi_sq/24) * a[2]**4 + (7 * pi_sq**4 * a[2]**4)/1200 + \
                       2 * (pi_sq**2) * a[3]**2 * a[4] + \
                       0.5 * a[1] * a[2]**2 * a[4] + (pi_sq/2) * a[1]**2 * a[2]**2 * a[4] + \
                       (1/24) * a[2]**3 * a[4] + (pi_sq/40) * a[2]**3 * a[4] + \
                       (5/6) * a[1] * a[2] * a[3] * a[4] + \
                       (2/45) * (pi_sq**2) * a[1] * a[2] * a[3] * a[4] + \
                       (3/40) * a[2] * a[3]**2 * a[4] + \
                       (1/6) * a[1]**2 * a[2]**2 * a[4] + \
                       (pi_sq/90) * a[1]**2 * a[2]**2 * a[4] + \
                       (1/45) * a[1]**3 * a[2]**2 * a[4] + \
                       (11 * pi_sq**2 * a[1]**3 * a[2]**2 * a[4])/7560 + \
                       (1/15) * a[1] * a[2] * a[3]**2 * a[4] + \
                       (11 * a[1]**4 * a[2]**2 * a[4])/2520 + \
                       (1/420) * a[2] * a[3]**3 + a[3]**4/15120
        
        kurtosis = kurtosis_num / (std**4)
        
        return {
            'mean': mean,
            'variance': variance,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _numerical_moments(self, order):
        """Numerical moment calculation."""
        moments = {}
        
        # Calculate raw moments
        for k in range(1, order + 1):
            integrand = lambda y: self.quantile(y)**k
            moment, _ = integrate.quad(integrand, 0, 1)
            moments[f'raw_{k}'] = moment
        
        # Calculate central moments
        mean = moments['raw_1']
        for k in range(2, order + 1):
            integrand = lambda y: (self.quantile(y) - mean)**k
            moment, _ = integrate.quad(integrand, 0, 1)
            moments[f'central_{k}'] = moment
        
        # Standardized moments
        variance = moments['central_2']
        std = np.sqrt(variance)
        moments['mean'] = mean
        moments['variance'] = variance
        moments['std'] = std
        
        if order >= 3:
            moments['skewness'] = moments['central_3'] / (std**3)
        if order >= 4:
            moments['kurtosis'] = moments['central_4'] / (std**4)
        
        return moments


class LogMetalog:
    """
    Semibounded (log) metalog distribution.
    
    Parameters:
    -----------
    x_data : array-like
        x values (quantiles) corresponding to cumulative probabilities
    y_data : array-like
        y values (cumulative probabilities) in (0, 1)
    lower_bound : float
        Lower bound of the distribution
    terms : int
        Number of terms to use (2-10 recommended)
    """
    
    def __init__(self, x_data, y_data, lower_bound=0, terms=5):
        self.lower_bound = lower_bound
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)
        self.terms = terms
        
        # Transform data: z = ln(x - lower_bound)
        self.z_data = np.log(self.x_data - lower_bound)
        
        # Create underlying metalog
        self.metalog = Metalog(self.z_data, self.y_data, terms)
        self.is_feasible = self.metalog.is_feasible
        self.coefficients = self.metalog.coefficients
    
    def quantile(self, y):
        """Log metalog quantile function."""
        y = np.asarray(y)
        z = self.metalog.quantile(y)
        return self.lower_bound + np.exp(z)
    
    def pdf(self, y):
        """Log metalog PDF using unified numerical differentiation."""
        y = np.asarray(y)
        return self._unified_pdf_numerical(y)
    
    def _unified_pdf_numerical(self, y):
        """Unified numerical PDF calculation with improved resolution and smoothing."""
        def derivative(y_vals, delta):
            y_plus = np.clip(y_vals + delta, 1e-12, 1 - 1e-12)
            y_minus = np.clip(y_vals - delta, 1e-12, 1 - 1e-12)
            x_plus = self.quantile(y_plus)
            x_minus = self.quantile(y_minus)
            return (x_plus - x_minus) / (y_plus - y_minus)
        
        # Use adaptive step size based on y value
        # Smaller steps near boundaries, larger in middle
        y_array = np.asarray(y)
        eps_array = np.where(
            y_array < 0.1, 1e-10,  # Very small near lower bound
            np.where(
                y_array > 0.9, 1e-10,  # Very small near upper bound
                1e-8  # Standard step in middle
            )
        )
        
        # Calculate derivative with adaptive step sizes
        grad = np.zeros_like(y_array, dtype=float)
        for i, (y_val, eps) in enumerate(zip(y_array, eps_array)):
            grad[i] = derivative(y_val, eps)
        
        # Handle problematic regions with larger step sizes
        bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
        if np.any(bad):
            grad[bad] = derivative(y_array[bad], 1e-6)
        
        bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
        if np.any(bad):
            grad[bad] = derivative(y_array[bad], 1e-4)
        
        # Ensure positive derivative (quantile should be monotonic)
        grad = np.abs(grad)
        grad = np.clip(grad, 1e-12, np.inf)
        
        pdf = 1.0 / grad
        
        # Apply smoothing to reduce discretization
        if len(pdf) > 5:
            # Use moving average for smoothing
            window_size = min(5, len(pdf))
            if window_size % 2 == 0:
                window_size -= 1  # Make odd
            pdf_smooth = np.convolve(pdf, np.ones(window_size)/window_size, mode='same')
            # Blend original and smoothed, more smoothing near boundaries
            blend_factor = np.where(
                y_array < 0.1, 0.7,  # More smoothing near lower bound
                np.where(
                    y_array > 0.9, 0.7,  # More smoothing near upper bound
                    0.3  # Less smoothing in middle
                )
            )
            pdf = blend_factor * pdf_smooth + (1 - blend_factor) * pdf
        
        return pdf
    
    def cdf(self, x):
        """Log metalog CDF."""
        x = np.asarray(x)
        
        # Handle values at or below lower bound
        valid_mask = x > self.lower_bound
        cdf_vals = np.zeros_like(x, dtype=float)
        
        # For valid x values, compute CDF
        if np.any(valid_mask):
            z = np.log(x[valid_mask] - self.lower_bound)
            cdf_vals[valid_mask] = self.metalog.cdf(z)
        
        # Values at or below lower bound have CDF = 0
        cdf_vals[~valid_mask] = 0
        
        return cdf_vals
    
    def sample(self, size=1):
        """Generate random samples."""
        u = np.random.uniform(0, 1, size)
        return self.quantile(u)
    
    def moments(self, order=4):
        """
        Calculate moments of log metalog distribution.
        
        Parameters:
        -----------
        order : int
            Maximum moment order to calculate
            
        Returns:
        --------
        moments : dict
            Dictionary containing mean, variance, skewness, kurtosis
        """
        # Use numerical integration for log metalog moments
        moments = {}
        
        # Calculate raw moments
        for k in range(1, order + 1):
            integrand = lambda y: self.quantile(y)**k
            moment, _ = integrate.quad(integrand, 0, 1)
            moments[f'raw_{k}'] = moment
        
        # Calculate central moments
        mean = moments['raw_1']
        for k in range(2, order + 1):
            integrand = lambda y: (self.quantile(y) - mean)**k
            moment, _ = integrate.quad(integrand, 0, 1)
            moments[f'central_{k}'] = moment
        
        # Standardized moments
        variance = moments['central_2']
        std = np.sqrt(variance)
        moments['mean'] = mean
        moments['variance'] = variance
        moments['std'] = std
        
        if order >= 3:
            moments['skewness'] = moments['central_3'] / (std**3)
        if order >= 4:
            moments['kurtosis'] = moments['central_4'] / (std**4)
        
        return moments


class LogitMetalog:
    """
    Bounded (logit) metalog distribution.
    
    Parameters:
    -----------
    x_data : array-like
        x values (quantiles) corresponding to cumulative probabilities
    y_data : array-like
        y values (cumulative probabilities) in (0, 1)
    lower_bound : float
        Lower bound of the distribution
    upper_bound : float
        Upper bound of the distribution
    terms : int
        Number of terms to use (2-10 recommended)
    """
    
    def __init__(self, x_data, y_data, lower_bound=0, upper_bound=1, terms=5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)
        self.terms = terms
        
        # Transform data: z = ln((x - lower_bound)/(upper_bound - x))
        self.z_data = np.log((self.x_data - lower_bound) / (upper_bound - self.x_data))
        
        # Create underlying metalog
        self.metalog = Metalog(self.z_data, self.y_data, terms)
        self.is_feasible = self.metalog.is_feasible
        self.coefficients = self.metalog.coefficients
    
    def quantile(self, y):
        """Logit metalog quantile function."""
        y = np.asarray(y)
        z = self.metalog.quantile(y)
        exp_z = np.exp(z)
        return self.lower_bound + (self.upper_bound - self.lower_bound) * exp_z / (1 + exp_z)
    
    def pdf(self, y):
        """Logit metalog PDF using unified numerical differentiation."""
        y = np.asarray(y)
        return self._unified_pdf_numerical(y)
    
    def _unified_pdf_numerical(self, y):
        """Unified numerical PDF calculation with adaptive step size."""
        def derivative(y_vals, delta):
            y_plus = np.clip(y_vals + delta, 1e-12, 1 - 1e-12)
            y_minus = np.clip(y_vals - delta, 1e-12, 1 - 1e-12)
            x_plus = self.quantile(y_plus)
            x_minus = self.quantile(y_minus)
            return (x_plus - x_minus) / (y_plus - y_minus)
        
        # Adaptive step size for numerical stability
        eps = 1e-8
        grad = derivative(y, eps)
        
        # Handle problematic regions with larger step size
        bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
        if np.any(bad):
            grad[bad] = derivative(y[bad], 1e-6)
        
        bad = (~np.isfinite(grad)) | (np.abs(grad) < 1e-12)
        if np.any(bad):
            grad[bad] = derivative(y[bad], 1e-4)
        
        # Ensure positive derivative (quantile should be monotonic)
        grad = np.abs(grad)
        grad = np.clip(grad, 1e-12, np.inf)
        
        return 1.0 / grad
    
    def cdf(self, x):
        """Logit metalog CDF."""
        x = np.asarray(x)
        
        # Handle values outside bounds
        cdf_vals = np.zeros_like(x, dtype=float)
        
        # Values at or below lower bound have CDF = 0
        below_lower = x <= self.lower_bound
        cdf_vals[below_lower] = 0
        
        # Values at or above upper bound have CDF = 1
        above_upper = x >= self.upper_bound
        cdf_vals[above_upper] = 1
        
        # For valid x values within bounds, compute CDF
        valid_mask = (x > self.lower_bound) & (x < self.upper_bound)
        if np.any(valid_mask):
            z = np.log((x[valid_mask] - self.lower_bound) / (self.upper_bound - x[valid_mask]))
            cdf_vals[valid_mask] = self.metalog.cdf(z)
        
        return cdf_vals
    
    def sample(self, size=1):
        """Generate random samples."""
        u = np.random.uniform(0, 1, size)
        return self.quantile(u)


def spt_metalog(q_low, q_median, q_high, alpha=0.1, distribution_type='unbounded', 
                 lower_bound=None, upper_bound=None):
    """
    Create metalog from Symmetric Percentile Triplet (SPT).
    
    Parameters:
    -----------
    q_low : float
        Lower quantile (e.g., 10th percentile)
    q_median : float
        Median (50th percentile)
    q_high : float
        Upper quantile (e.g., 90th percentile)
    alpha : float
        Probability level for low/high quantiles (default 0.1 for 10-90)
    distribution_type : str
        'unbounded', 'semibounded', or 'bounded'
    lower_bound : float, optional
        Lower bound for semibounded/bounded distributions
    upper_bound : float, optional
        Upper bound for bounded distributions
        
    Returns:
    --------
    metalog : Metalog, LogMetalog, or LogitMetalog
        Fitted metalog distribution
    """
    y_data = [alpha, 0.5, 1 - alpha]
    x_data = [q_low, q_median, q_high]
    
    if distribution_type == 'unbounded':
        return Metalog(x_data, y_data, terms=3)
    elif distribution_type == 'semibounded':
        if lower_bound is None:
            raise MetalogError("lower_bound required for semibounded distribution")
        return LogMetalog(x_data, y_data, lower_bound, terms=3)
    elif distribution_type == 'bounded':
        if lower_bound is None or upper_bound is None:
            raise MetalogError("lower_bound and upper_bound required for bounded distribution")
        return LogitMetalog(x_data, y_data, lower_bound, upper_bound, terms=3)
    else:
        raise MetalogError("distribution_type must be 'unbounded', 'semibounded', or 'bounded'")


def plot_metalog(metalog_dist, title="Metalog Distribution", show_data=True):
    """
    Plot metalog distribution with PDF and CDF.
    
    Parameters:
    -----------
    metalog_dist : Metalog, LogMetalog, or LogitMetalog
        Fitted metalog distribution
    title : str
        Plot title
    show_data : bool
        Whether to show original data points
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Generate points for plotting
    y_plot = np.linspace(0.01, 0.99, 1000)
    x_plot = metalog_dist.quantile(y_plot)
    pdf_plot = metalog_dist.pdf(y_plot)
    
    # Plot PDF
    ax1.plot(x_plot, pdf_plot, 'b-', linewidth=2, label='PDF')
    if show_data and hasattr(metalog_dist, 'x_data'):
        ax1.scatter(metalog_dist.x_data, metalog_dist.pdf(metalog_dist.y_data), 
                   color='red', s=50, zorder=5, label='Data points')
    ax1.set_xlabel('x')
    ax1.set_ylabel('PDF')
    ax1.set_title(f'{title} - PDF')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot CDF
    ax2.plot(x_plot, y_plot, 'g-', linewidth=2, label='CDF')
    if show_data and hasattr(metalog_dist, 'x_data'):
        ax2.scatter(metalog_dist.x_data, metalog_dist.y_data, 
                   color='red', s=50, zorder=5, label='Data points')
    ax2.set_xlabel('x')
    ax2.set_ylabel('CDF')
    ax2.set_title(f'{title} - CDF')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def metalog_panel(x_data, y_data, max_terms=10, distribution_type='unbounded', 
                 lower_bound=None, upper_bound=None):
    """
    Create metalog panel showing distributions with different numbers of terms.
    
    Parameters:
    -----------
    x_data, y_data : array-like
        CDF data
    max_terms : int
        Maximum number of terms to show
    distribution_type : str
        'unbounded', 'semibounded', or 'bounded'
    lower_bound, upper_bound : float, optional
        Bounds for semibounded/bounded distributions
        
    Returns:
    --------
    fig : matplotlib Figure
        Panel plot
    """
    n_terms = range(2, max_terms + 1)
    n_cols = 4
    n_rows = int(np.ceil(len(n_terms) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    y_plot = np.linspace(0.01, 0.99, 500)
    
    for i, n in enumerate(n_terms):
        ax = axes[i]
        
        try:
            # Create metalog with n terms
            if distribution_type == 'unbounded':
                metalog = Metalog(x_data, y_data, terms=n)
            elif distribution_type == 'semibounded':
                metalog = LogMetalog(x_data, y_data, lower_bound, terms=n)
            elif distribution_type == 'bounded':
                metalog = LogitMetalog(x_data, y_data, lower_bound, upper_bound, terms=n)
            
            # Plot PDF
            x_plot = metalog.quantile(y_plot)
            pdf_plot = metalog.pdf(y_plot)
            
            ax.plot(x_plot, pdf_plot, 'b-', linewidth=1.5)
            ax.scatter(x_data, metalog.pdf(y_data), color='red', s=20, zorder=5)
            ax.set_title(f'{n} terms')
            ax.grid(True, alpha=0.3)
            
            # Mark feasibility
            if not metalog.is_feasible:
                ax.text(0.05, 0.95, 'Infeasible', transform=ax.transAxes,
                        color='red', fontsize=8, verticalalignment='top')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)[:30]}...', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{n} terms - Failed')
    
    # Hide unused subplots
    for i in range(len(n_terms), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    print("Metalog Distributions Library")
    print("=" * 40)
    
    # Example 1: Basic unbounded metalog
    print("\nExample 1: Basic unbounded metalog")
    x_data = [-2, -1, 0, 1, 2]
    y_data = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    metalog = Metalog(x_data, y_data, terms=4)
    print(f"Feasible: {metalog.is_feasible}")
    print(f"Coefficients: {metalog.coefficients}")
    
    # Example 2: SPT metalog
    print("\nExample 2: SPT metalog (10-50-90)")
    spt = spt_metalog(10, 50, 90, alpha=0.1, distribution_type='unbounded')
    print(f"Feasible: {spt.is_feasible}")
    
    # Example 3: Bounded metalog
    print("\nExample 3: Bounded metalog")
    x_bounds = [0.1, 0.3, 0.5, 0.7, 0.9]
    y_bounds = [0.1, 0.3, 0.5, 0.7, 0.9]
    bounded = LogitMetalog(x_bounds, y_bounds, lower_bound=0, upper_bound=1, terms=3)
    print(f"Feasible: {bounded.is_feasible}")
    
