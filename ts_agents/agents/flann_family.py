"""
agents/flann_family.py
══════════════════════
Pure-numpy implementation of the FLANN family for time-series forecasting.

Three variants
──────────────
  FLANN         – Functional Link Artificial Neural Network (Pao, 1989)
                  Expands inputs into a non-linear basis (polynomial,
                  trigonometric, or mixed), then solves a single linear
                  output layer via ridge regression.  No hidden layer,
                  no back-prop.  O(n·D²) training.

  RecurrentFLANN – FLANN with a recurrent state: at each step the
                  previous prediction (or a small feedback window) is
                  appended to the input before basis expansion.  This
                  captures temporal dynamics that pure lag features miss.
                  Training uses teacher-forcing on the training window;
                  forecasting uses its own recursive predictions as the
                  recurrent inputs.

  RVFL           – Random Vector Functional Link (Pao & Takefuji, 1994)
                  Like FLANN but the hidden-layer weights are drawn once
                  from a random distribution and FIXED.  Only the output
                  weights are learned (ridge regression on the
                  random-projected features).  Consistently fast and
                  surprisingly competitive, especially with small n.

All three share the same interface:
  model = FLANNFamily(variant=..., **kwargs)
  model.fit(X_train, y_train)
  preds = model.predict_multi_step(X_seed, horizon)  # iterative future
  preds = model.predict(X_test)                       # one-step

Basis families
──────────────
  "polynomial"     – [x, x², x³, …, xᴺ]
  "trigonometric"  – [sin(πx), cos(πx), sin(2πx), cos(2πx), …]
  "chebyshev"      – Chebyshev polynomials T₁(x)…Tₙ(x)
  "mixed"          – polynomial ∪ trigonometric  (default)
  "legendre"       – Legendre polynomials P₁(x)…Pₙ(x)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Literal, Optional, Tuple, Any


# ── type alias ────────────────────────────────────────────────────────────────
BasisFamily = Literal[
    "polynomial", "trigonometric", "chebyshev", "mixed", "legendre"
]

SUPPORTED_VARIANTS = ("flann", "recurrent_flann", "rvfl")
SUPPORTED_BASIS    = ("polynomial", "trigonometric", "chebyshev", "mixed", "legendre")


# ─────────────────────────────────────────────────────────────────────────────
# BASIS EXPANSION
# ─────────────────────────────────────────────────────────────────────────────

def _chebyshev_T(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate Chebyshev polynomials T₁…T_order at every element of x."""
    parts = []
    T_prev = np.ones_like(x)    # T₀
    T_curr = x.copy()           # T₁
    parts.append(T_curr)
    for k in range(2, order + 1):
        T_next = 2.0 * x * T_curr - T_prev
        parts.append(T_next)
        T_prev, T_curr = T_curr, T_next
    return np.stack(parts, axis=-1)   # (..., order)


def _legendre_P(x: np.ndarray, order: int) -> np.ndarray:
    """Evaluate Legendre polynomials P₁…P_order (clipped to [-1,1])."""
    xc = np.clip(x, -1.0, 1.0)
    parts = []
    P_prev = np.ones_like(xc)   # P₀
    P_curr = xc.copy()          # P₁
    parts.append(P_curr)
    for k in range(2, order + 1):
        P_next = ((2*k - 1) * xc * P_curr - (k - 1) * P_prev) / k
        parts.append(P_next)
        P_prev, P_curr = P_curr, P_next
    return np.stack(parts, axis=-1)


def expand_basis(
    X: np.ndarray,
    order: int,
    family: BasisFamily,
    mean_: np.ndarray,
    std_: np.ndarray,
) -> np.ndarray:
    """
    Expand a 2-D feature matrix X (n_samples × n_features) into the
    chosen non-linear basis, then prepend a bias column.

    Returns shape (n_samples, 1 + expanded_features).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n, d = X.shape

    # Normalize each feature to roughly [-1, 1] for stable expansions
    scale = np.where(np.abs(std_) < 1e-8, 1.0, std_)
    Xn = np.clip((X - mean_) / scale, -4.0, 4.0)

    parts: List[np.ndarray] = []

    if family in ("polynomial", "mixed"):
        for p in range(1, order + 1):
            parts.append(Xn ** p)

    if family in ("trigonometric", "mixed"):
        for k in range(1, order + 1):
            parts.append(np.sin(k * np.pi * Xn))
            parts.append(np.cos(k * np.pi * Xn))

    if family == "chebyshev":
        # Apply Chebyshev expansion to each feature independently
        xc = np.clip(Xn, -1.0, 1.0)
        for col in range(d):
            cheb = _chebyshev_T(xc[:, col], order)   # (n, order)
            parts.append(cheb)

    if family == "legendre":
        xc = np.clip(Xn, -1.0, 1.0)
        for col in range(d):
            leg = _legendre_P(xc[:, col], order)     # (n, order)
            parts.append(leg)

    if not parts:
        # fallback: just use raw normalized inputs
        parts.append(Xn)

    Z = np.hstack(parts)   # (n, total_expanded_features)
    bias = np.ones((n, 1), dtype=float)
    return np.hstack([bias, Z])   # prepend bias


# ─────────────────────────────────────────────────────────────────────────────
# RIDGE SOLVE
# ─────────────────────────────────────────────────────────────────────────────

def _ridge_solve(Z: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve the ridge-regularised normal equations:  w = (ZᵀZ + λI)⁻¹ Zᵀy."""
    lam = max(1e-12, float(lam))
    eye = np.eye(Z.shape[1], dtype=float)
    eye[0, 0] = 0.0   # leave bias unregularised
    A = Z.T @ Z + lam * eye
    b = Z.T @ y.reshape(-1)
    try:
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w


# ─────────────────────────────────────────────────────────────────────────────
# FLANNFamily – unified interface
# ─────────────────────────────────────────────────────────────────────────────

class FLANNFamily:
    """
    Unified interface for the FLANN / RecurrentFLANN / RVFL family.

    Parameters
    ----------
    variant        : "flann" | "recurrent_flann" | "rvfl"
    basis_family   : "polynomial" | "trigonometric" | "chebyshev"
                     | "mixed" | "legendre"
    order          : expansion order (polynomial degree or trig harmonics)
    ridge_lambda   : L2 regularisation for the output weights
    recurrent_depth: (RecurrentFLANN only) number of past predictions fed back
    rvfl_n_hidden  : (RVFL only) number of random hidden nodes
    rvfl_activation: (RVFL only) "sigmoid" | "relu" | "tanh" | "sin"
    rvfl_seed      : random seed for reproducibility
    """

    def __init__(
        self,
        variant: str = "flann",
        basis_family: BasisFamily = "mixed",
        order: int = 3,
        ridge_lambda: float = 1e-2,
        recurrent_depth: int = 1,
        rvfl_n_hidden: int = 64,
        rvfl_activation: str = "sigmoid",
        rvfl_seed: int = 42,
    ):
        variant = variant.lower().strip().replace("-", "_")
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. "
                f"Choose from: {SUPPORTED_VARIANTS}"
            )
        if basis_family not in SUPPORTED_BASIS:
            raise ValueError(
                f"Unknown basis_family '{basis_family}'. "
                f"Choose from: {SUPPORTED_BASIS}"
            )

        self.variant        = variant
        self.basis_family   = basis_family
        self.order          = max(1, int(order))
        self.ridge_lambda   = float(ridge_lambda)
        self.recurrent_depth = max(1, int(recurrent_depth))
        self.rvfl_n_hidden  = max(4, int(rvfl_n_hidden))
        self.rvfl_activation = rvfl_activation.lower().strip()
        self.rvfl_seed      = int(rvfl_seed)

        # Learned/fixed attributes (set during fit)
        self._mean: Optional[np.ndarray]    = None
        self._std:  Optional[np.ndarray]    = None
        self._weights: Optional[np.ndarray] = None
        self._n_input_features: int         = 0
        self._fitted: bool                  = False

        # RVFL-specific: fixed random projection matrix
        self._W_random: Optional[np.ndarray] = None
        self._b_random: Optional[np.ndarray] = None

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "FLANNFamily":
        """Fit the model. X_train is (n_samples, n_features); y_train is (n_samples,)."""
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=float).reshape(-1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._mean = np.nanmean(X, axis=0)
        self._std  = np.nanstd(X,  axis=0)
        self._n_input_features = X.shape[1]

        if self.variant == "rvfl":
            Z = self._rvfl_project(X, fit=True)
        elif self.variant == "recurrent_flann":
            # Training with teacher-forcing: append recurrent targets to X
            X_rec = self._build_recurrent_X(X, y_teacher=y)
            Z = expand_basis(X_rec, self.order, self.basis_family,
                             self._mean_rec, self._std_rec)
        else:  # plain FLANN
            Z = expand_basis(X, self.order, self.basis_family,
                             self._mean, self._std)

        self._weights = _ridge_solve(Z, y, self.ridge_lambda)
        self._fitted  = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """One-step-ahead prediction for each row in X."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.variant == "rvfl":
            Z = self._rvfl_project(X, fit=False)
        else:
            mean = getattr(self, "_mean_rec", self._mean)
            std  = getattr(self, "_std_rec",  self._std)
            if self.variant == "recurrent_flann":
                # At inference time without teacher, use zero recurrent state
                n = X.shape[0]
                rec_pad = np.zeros((n, self.recurrent_depth), dtype=float)
                X_aug = np.hstack([X, rec_pad])
                Z = expand_basis(X_aug, self.order, self.basis_family, mean, std)
            else:
                Z = expand_basis(X, self.order, self.basis_family, mean, std)
        return (Z @ self._weights).reshape(-1)

    def predict_multi_step(
        self,
        X_seed: np.ndarray,
        horizon: int,
        lag_indices: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Iterative multi-step forecast.

        For FLANN/RVFL:  the forecast at step h uses actual predictions from
            h-1, h-2, … as lag features (specified by lag_indices, which are
            column indices in X_seed that correspond to lag features).

        For RecurrentFLANN:  previous predictions are fed back as the
            recurrent state vector AND the lag columns are also updated.

        Parameters
        ----------
        X_seed      : the last known row of X (1, n_features) — used as the
                      starting template for future rows
        horizon     : number of future steps to predict
        lag_indices : list of column indices in X that are lag features
                      (index 0 = lag-1, index 1 = lag-2, etc.)
                      If None, first min(3, n_features) columns are assumed.

        Returns
        -------
        np.ndarray of shape (horizon,) with forecasted values.
        """
        self._check_fitted()
        X_seed = np.asarray(X_seed, dtype=float)
        if X_seed.ndim == 1:
            X_seed = X_seed.reshape(1, -1)

        if lag_indices is None:
            lag_indices = list(range(min(3, self._n_input_features)))

        preds:    List[float] = []
        rec_buf:  List[float] = [0.0] * self.recurrent_depth

        x_row = X_seed.copy()   # shape (1, n_features)

        for _ in range(horizon):
            if self.variant == "rvfl":
                Z = self._rvfl_project(x_row, fit=False)
                pred = float((Z @ self._weights).reshape(-1)[0])
            elif self.variant == "recurrent_flann":
                mean = getattr(self, "_mean_rec", self._mean)
                std  = getattr(self, "_std_rec",  self._std)
                rec_vec = np.array(rec_buf, dtype=float).reshape(1, -1)
                x_aug = np.hstack([x_row, rec_vec])
                Z = expand_basis(x_aug, self.order, self.basis_family, mean, std)
                pred = float((Z @ self._weights).reshape(-1)[0])
                # Update recurrent buffer (FIFO)
                rec_buf = [pred] + rec_buf[:-1]
            else:  # FLANN
                Z = expand_basis(x_row, self.order, self.basis_family,
                                 self._mean, self._std)
                pred = float((Z @ self._weights).reshape(-1)[0])

            preds.append(pred)

            # Shift lag features: lag-1 gets last prediction, lag-2 gets
            # old lag-1, etc.
            x_new = x_row.copy()
            for rank, col_idx in enumerate(lag_indices):
                if col_idx >= x_new.shape[1]:
                    continue
                if rank == 0:
                    x_new[0, col_idx] = pred
                else:
                    prev_col = lag_indices[rank - 1]
                    if prev_col < x_new.shape[1]:
                        x_new[0, col_idx] = x_row[0, prev_col]
            x_row = x_new

        return np.array(preds, dtype=float)

    # ── RVFL internals ────────────────────────────────────────────────────────

    def _rvfl_project(self, X: np.ndarray, fit: bool) -> np.ndarray:
        """
        Random projection layer for RVFL.
        Hidden output: h_i = activation(Wᵢ·x + bᵢ) for i = 1…H
        Final feature vector: [x (direct link), h₁, …, h_H, bias]
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n, d = X.shape

        # Normalize
        scale = np.where(np.abs(self._std) < 1e-8, 1.0, self._std)
        Xn = np.clip((X - self._mean) / scale, -4.0, 4.0)

        if fit or self._W_random is None:
            rng = np.random.default_rng(self.rvfl_seed)
            # Xavier / Glorot initialisation
            limit = np.sqrt(6.0 / (d + self.rvfl_n_hidden))
            self._W_random = rng.uniform(-limit, limit,
                                         (d, self.rvfl_n_hidden)).astype(float)
            self._b_random = rng.uniform(-limit, limit,
                                         (self.rvfl_n_hidden,)).astype(float)

        pre_act = Xn @ self._W_random + self._b_random   # (n, H)

        act = self.rvfl_activation
        if act == "sigmoid":
            H = 1.0 / (1.0 + np.exp(-np.clip(pre_act, -30, 30)))
        elif act == "relu":
            H = np.maximum(0.0, pre_act)
        elif act == "tanh":
            H = np.tanh(pre_act)
        elif act == "sin":
            H = np.sin(pre_act)
        else:
            H = np.tanh(pre_act)

        bias = np.ones((n, 1), dtype=float)
        # Direct links (original normalized inputs) + random hidden layer + bias
        return np.hstack([Xn, H, bias])

    # ── RecurrentFLANN internals ──────────────────────────────────────────────

    def _build_recurrent_X(
        self, X: np.ndarray, y_teacher: np.ndarray
    ) -> np.ndarray:
        """
        Build teacher-forced recurrent feature matrix.
        Appends [y_{t-1}, y_{t-2}, …, y_{t-depth}] to each row of X.
        """
        n = X.shape[0]
        y_teacher = np.asarray(y_teacher, dtype=float)
        # y_teacher may be longer than X (e.g. because X was dropna'd)
        # align to the last n elements
        if len(y_teacher) > n:
            y_teacher = y_teacher[-n:]
        elif len(y_teacher) < n:
            y_teacher = np.concatenate([np.zeros(n - len(y_teacher)), y_teacher])

        rec = np.zeros((n, self.recurrent_depth), dtype=float)
        for i in range(n):
            for lag in range(1, self.recurrent_depth + 1):
                src_idx = i - lag
                if src_idx >= 0:
                    rec[i, lag - 1] = float(y_teacher[src_idx])
                # else: leave as 0 (start-of-sequence padding)

        X_aug = np.hstack([X, rec])
        # Recompute normalisation stats for augmented feature vector
        self._mean_rec = np.nanmean(X_aug, axis=0)
        self._std_rec  = np.nanstd(X_aug,  axis=0)
        return X_aug

    # ── misc ─────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self._fitted or self._weights is None:
            raise RuntimeError("FLANNFamily model is not fitted yet.")

    @property
    def n_params(self) -> int:
        """Number of output-layer parameters."""
        if self._weights is not None:
            return int(self._weights.shape[0])
        return 0

    def summary(self) -> Dict[str, Any]:
        return {
            "variant":         self.variant,
            "basis_family":    self.basis_family,
            "order":           self.order,
            "ridge_lambda":    self.ridge_lambda,
            "recurrent_depth": self.recurrent_depth if self.variant == "recurrent_flann" else None,
            "rvfl_n_hidden":   self.rvfl_n_hidden   if self.variant == "rvfl"            else None,
            "rvfl_activation": self.rvfl_activation  if self.variant == "rvfl"            else None,
            "n_output_params": self.n_params,
            "fitted":          self._fitted,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"FLANNFamily(variant={s['variant']}, basis={s['basis_family']}, "
            f"order={s['order']}, lambda={s['ridge_lambda']}, "
            f"fitted={s['fitted']})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE RUNNER  (mirrors _run_flann_forecast in server.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_flann_family_forecast(
    y_full: "pd.Series",
    y_train: "pd.Series",
    y_test:  "pd.Series",
    future_idx: "pd.Index",
    exog_all:   Optional["pd.DataFrame"],
    period_used: int,
    variant:      str = "flann",
    basis_family: str = "mixed",
    order:        int = 3,
    ridge_lambda: float = 1e-2,
    recurrent_depth: int = 1,
    rvfl_n_hidden:   int = 64,
    rvfl_activation: str = "sigmoid",
) -> Optional[Dict[str, np.ndarray]]:
    """
    Train a FLANNFamily model on y_train (with lag/exog features),
    evaluate on y_test (holdout), and produce future forecasts.

    Returns {"holdout_pred": ndarray, "future_pred": ndarray}
    or None if there is insufficient data.
    """
    import pandas as pd

    y_full  = pd.Series(y_full).dropna().astype(float)
    y_train = pd.Series(y_train).dropna().astype(float)
    y_test  = pd.Series(y_test).dropna().astype(float)

    if len(y_train) < 8 or len(y_test) == 0:
        return None

    p = int(period_used) if period_used and int(period_used) > 1 else 0
    lag_candidates = sorted({1, 2, 3, p, p + 1} - {0} if p > 1
                            else {1, 2, 3})
    lag_candidates = [lag for lag in lag_candidates if lag < len(y_full)]
    if not lag_candidates:
        return None

    # Build feature DataFrame
    fl_df = pd.DataFrame(index=y_full.index)
    fl_df["y"] = y_full.values
    for lag in lag_candidates:
        fl_df[f"lag{lag}"] = fl_df["y"].shift(lag)
    if len(y_full) >= 4:
        fl_df["roll_mean_3"] = fl_df["y"].shift(1).rolling(3).mean()
        fl_df["roll_std_3"]  = fl_df["y"].shift(1).rolling(3).std()
    if len(y_full) >= 8:
        fl_df["roll_mean_7"] = fl_df["y"].shift(1).rolling(7).mean()
        fl_df["roll_std_7"]  = fl_df["y"].shift(1).rolling(7).std()

    ex_cols: List[str] = []
    if exog_all is not None and len(exog_all.columns):
        for c in exog_all.columns:
            sn = f"exog__{c}"
            fl_df[sn] = exog_all[c].reindex(fl_df.index).astype(float).values
            ex_cols.append(sn)

    fl_df = fl_df.dropna()
    if len(fl_df) < 8:
        return None

    train_cut = y_train.index[-1]
    fl_train = fl_df[fl_df.index <= train_cut]
    fl_test  = fl_df[fl_df.index.isin(y_test.index)]
    if len(fl_train) < 5 or len(fl_test) == 0:
        return None

    x_cols = [c for c in fl_df.columns if c != "y"]
    # Lag feature column indices (0-based in x_cols)
    lag_col_indices = [i for i, c in enumerate(x_cols)
                       if c.startswith("lag")]

    model = FLANNFamily(
        variant=variant,
        basis_family=basis_family,
        order=order,
        ridge_lambda=ridge_lambda,
        recurrent_depth=recurrent_depth,
        rvfl_n_hidden=rvfl_n_hidden,
        rvfl_activation=rvfl_activation,
    )
    model.fit(fl_train[x_cols].values, fl_train["y"].values)

    # Holdout predictions (one-step each row, not recursive)
    yhat_test = model.predict(fl_test[x_cols].values)

    # Future: iterative multi-step
    seed_row = fl_df.iloc[[-1]][x_cols].values   # last observed row
    fut_preds = model.predict_multi_step(
        X_seed=seed_row,
        horizon=len(future_idx),
        lag_indices=lag_col_indices,
    )

    return {
        "holdout_pred": np.asarray(yhat_test, dtype=float),
        "future_pred":  np.asarray(fut_preds,  dtype=float),
        "model_summary": model.summary(),
    }
