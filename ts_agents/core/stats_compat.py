"""
core/stats_compat.py
─────────────────────
Pure numpy/scipy implementations of statistical tests and decompositions
used as a drop-in replacement when statsmodels is not available.

Covered:
  - ADF test (augmented Dickey-Fuller via OLS)
  - KPSS test (variance-ratio proxy)
  - STL decomposition (LOESS-based via scipy)
  - HP filter (Hodrick-Prescott)
  - ACF / PACF
  - Ljung-Box Q-test
  - ARCH-LM test
  - Theil-Sen slope
"""
from __future__ import annotations
import numpy as np
from scipy import signal as sp_signal
from scipy.stats import t as t_dist, shapiro, theilslopes
from typing import Dict, List, Tuple, Optional


# ── ADF test (simplified, no lag optimisation) ───────────────────────────────

def adf_test(y: np.ndarray, maxlag: int = 1) -> Dict:
    """
    Simplified ADF: regress Δy_t on y_{t-1} and lagged differences.
    Returns statistic, p-value (approximated), and stationarity flag.
    """
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    n = len(dy)

    # build regressor matrix
    X_cols = [y[:-1]]  # lagged level
    for lag in range(1, min(maxlag + 1, n // 3)):
        if lag < n:
            padded = np.concatenate([np.full(lag, np.nan), dy[:-lag]])
            X_cols.append(padded)
    X = np.column_stack(X_cols)
    X = np.column_stack([np.ones(n), X])

    # drop NaN rows
    mask = ~np.isnan(X).any(axis=1)
    X, y_reg = X[mask], dy[mask]

    try:
        beta = np.linalg.lstsq(X, y_reg, rcond=None)[0]
        resid = y_reg - X @ beta
        sigma2 = np.sum(resid**2) / max(1, len(y_reg) - X.shape[1])
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(max(0, cov[1, 1]))
        stat = beta[1] / (se + 1e-12)
    except Exception:
        stat = 0.0

    # MacKinnon approximate critical values (no-trend case)
    # p-value approximation using t-distribution (conservative)
    df = max(1, len(y_reg) - X.shape[1])
    p_value = float(t_dist.cdf(stat, df=df))  # one-tailed left
    p_value = min(1.0, max(0.001, p_value))

    # Rough critical values from MacKinnon tables
    crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    stationary = stat < crit["5%"]

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "stationary": stationary,
        "critical_values": crit,
    }


def kpss_test(y: np.ndarray) -> Dict:
    """
    KPSS test: H0 = stationary.
    Uses the variance-ratio formulation.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    mu = np.mean(y)
    e = y - mu
    S = np.cumsum(e)
    s2 = np.sum(e**2) / n
    stat = float(np.sum(S**2) / (n**2 * (s2 + 1e-12)))

    # Approximate critical values (level, no trend)
    crit = {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739}
    if stat < crit["10%"]:
        p_value = 0.20
    elif stat < crit["5%"]:
        p_value = 0.10
    elif stat < crit["2.5%"]:
        p_value = 0.05
    elif stat < crit["1%"]:
        p_value = 0.025
    else:
        p_value = 0.01

    return {
        "statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "stationary": p_value > 0.05,   # KPSS: fail to reject H0 ⟹ stationary
    }


# ── STL-like decomposition (LOESS via scipy) ──────────────────────────────────

def stl_decompose(
    y: np.ndarray,
    period: int,
    robust: bool = True,
    n_iter: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Simplified iterative STL:
    1. Estimate trend with LOESS (Savitzky-Golay as proxy)
    2. Detrend → extract seasonal via period averaging
    3. Residual = original - trend - seasonal
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # LOESS proxy: Savitzky-Golay filter
    window = min(max(period * 2 + 1, 7), n - (1 if n % 2 == 0 else 0))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)

    trend = _savgol_trend(y, window)
    detrended = y - trend

    # seasonal: average each phase position across all cycles
    seasonal = np.zeros(n)
    for phase in range(period):
        positions = np.arange(phase, n, period)
        vals = detrended[positions]
        if robust:
            med = np.nanmedian(vals)
            seasonal[positions] = med
        else:
            seasonal[positions] = np.nanmean(vals)

    # centre seasonal (so it sums to 0 per cycle)
    seasonal -= np.nanmean(seasonal)

    residual = y - trend - seasonal
    return {"trend": trend, "seasonal": seasonal, "residual": residual}


def _savgol_trend(y: np.ndarray, window: int, polyorder: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing as LOESS proxy."""
    from scipy.signal import savgol_filter
    n = len(y)
    window = min(window, n)
    if window % 2 == 0:
        window -= 1
    window = max(window, polyorder + 2)
    if window > n:
        return np.full(n, np.nanmean(y))
    try:
        return savgol_filter(y, window_length=window, polyorder=min(polyorder, window - 1))
    except Exception:
        return np.full(n, np.nanmean(y))


# ── Hodrick-Prescott filter ───────────────────────────────────────────────────

def hp_filter(y: np.ndarray, lamb: float = 1600.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    HP filter via the matrix approach:
      min_g Σ(y-g)² + λ Σ(Δ²g)²
    Solved as: (I + λ F'F) g = y
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    # The dense matrix form is O(n^2) in memory and becomes impossible for
    # long series (for example 100k+ observations). Fall back to a smooth
    # trend proxy instead of trying to allocate an n x n system.
    if n > 5000:
        window = min(1001, n if n % 2 == 1 else n - 1)
        window = max(7, window)
        trend = _savgol_trend(y, window)
        cycle = y - trend
        return cycle, trend

    I = np.eye(n)
    # Second difference matrix
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1
        D[i, i + 1] = -2
        D[i, i + 2] = 1
    A = I + lamb * D.T @ D
    try:
        trend = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        trend = y.copy()
    cycle = y - trend
    return cycle, trend


# ── ACF / PACF ────────────────────────────────────────────────────────────────

def acf(y: np.ndarray, nlags: int = 40) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y_c = y - np.mean(y)
    n = len(y_c)
    denom = np.dot(y_c, y_c)
    result = [1.0]
    for k in range(1, min(nlags + 1, n)):
        result.append(float(np.dot(y_c[k:], y_c[:-k]) / (denom + 1e-12)))
    return np.array(result)


def pacf(y: np.ndarray, nlags: int = 40) -> np.ndarray:
    """PACF via Yule-Walker equations."""
    acf_vals = acf(y, nlags=nlags)
    n_lags = len(acf_vals) - 1
    pacf_vals = [1.0]
    phi = np.zeros((n_lags, n_lags))
    for k in range(1, n_lags + 1):
        if k == 1:
            phi[0, 0] = acf_vals[1]
        else:
            denom = 1 - sum(phi[k-2, j] * acf_vals[k-1-j] for j in range(k-1))
            if abs(denom) < 1e-12:
                phi[k-1, k-1] = 0.0
            else:
                num = acf_vals[k] - sum(phi[k-2, j] * acf_vals[k-1-j] for j in range(k-1))
                phi[k-1, k-1] = num / denom
                for j in range(k-1):
                    phi[k-1, j] = phi[k-2, j] - phi[k-1, k-1] * phi[k-2, k-2-j]
        pacf_vals.append(float(phi[k-1, k-1]))
    return np.array(pacf_vals[:n_lags+1])


# ── Ljung-Box Q test ─────────────────────────────────────────────────────────

def ljungbox(y: np.ndarray, lags: int = 10) -> Dict:
    y = np.asarray(y, dtype=float)
    n = len(y)
    acf_vals = acf(y, nlags=lags)
    Q = float(n * (n + 2) * np.sum(
        [acf_vals[k]**2 / (n - k) for k in range(1, lags + 1)]
    ))
    from scipy.stats import chi2
    p_value = float(1 - chi2.cdf(Q, df=lags))
    return {"lb_stat": round(Q, 4), "lb_pvalue": round(p_value, 4)}


# ── ARCH-LM test ──────────────────────────────────────────────────────────────

def arch_lm(resid: np.ndarray, nlags: int = 5) -> Dict:
    """ARCH-LM: regress squared residuals on lagged squared residuals."""
    r = np.asarray(resid, dtype=float)
    r2 = r**2
    n = len(r2)
    X = np.column_stack([r2[i:n-nlags+i] for i in range(nlags)] + [np.ones(n - nlags)])
    y_r = r2[nlags:]
    try:
        beta = np.linalg.lstsq(X, y_r, rcond=None)[0]
        yhat = X @ beta
        ss_res = np.sum((y_r - yhat)**2)
        ss_tot = np.sum((y_r - np.mean(y_r))**2)
        r2_score = 1 - ss_res / (ss_tot + 1e-12)
        lm_stat = float(n * r2_score)
        from scipy.stats import chi2
        p_value = float(1 - chi2.cdf(lm_stat, df=nlags))
    except Exception:
        lm_stat, p_value = 0.0, 1.0
    return {"lm_stat": round(lm_stat, 4), "lm_p": round(p_value, 4)}


# ── OLS ───────────────────────────────────────────────────────────────────────

class OLSResult:
    def __init__(self, params, pvalues, rsquared):
        self.params = params
        self.pvalues = pvalues
        self.rsquared = rsquared

def ols_fit(X: np.ndarray, y: np.ndarray) -> OLSResult:
    """Minimal OLS with t-test p-values."""
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        n, k = X.shape
        sigma2 = np.sum(resid**2) / max(1, n - k)
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.maximum(0, np.diag(cov)))
        t_stats = beta / (se + 1e-12)
        p_vals = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=max(1, n - k)))
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
    except Exception:
        k = X.shape[1] if X.ndim > 1 else 1
        beta = np.zeros(k)
        p_vals = np.ones(k)
        r2 = 0.0
    return OLSResult(beta, p_vals, r2)
