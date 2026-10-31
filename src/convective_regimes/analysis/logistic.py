from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SlidingGammaResult:
    """Single-window output of the sliding logistic regression."""

    gamma_center: float

    beta_ver: float
    beta_hor: float
    intercept: float

    auc_combined: float
    auc_bver_only: float
    auc_bhor_only: float

    n_samples: int
    positive_fraction: float


def fit_logistic_model(
        X: np.ndarray,
        y: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Fit logistic regression and return coefficients in original feature units.

    Parameters
    ----------
    X : shape (n_samples, n_features)
    y : binary target array

    Returns
    -------
    coef : coefficients rescaled to original units
    intercept : intercept rescaled to original units
    auc : ROC-AUC evaluated on the training set
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=5000)
    model.fit(Xs, y)

    prob = model.predict_proba(Xs)[:, 1]
    auc = roc_auc_score(y, prob)

    coef = model.coef_[0] / scaler.scale_
    intercept = (
            model.intercept_[0]
            - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_)
    )

    return coef, intercept, auc


def sliding_gamma_analysis(
        *,
        gamma: np.ndarray,
        bver: np.ndarray,
        bhor: np.ndarray,
        target: np.ndarray,
        gamma_centers: Sequence[float] | None = None,
        window_half_width: float = 5.0,
        min_samples: int = 1000,
) -> pd.DataFrame:
    """
    Sliding-window logistic regression in inclination space.

    For each window centred on a gamma value, fits a combined (bver + bhor)
    model and two single-variable models, recording coefficients and AUC scores.

    Parameters
    ----------
    gamma : inclination in degrees
    bver : vertical magnetic field [G]
    bhor : horizontal magnetic field [G]
    target : binary label (1 = positive class, 0 = negative)
    gamma_centers : window centres; defaults to np.arange(5, 86, 2)
    window_half_width : half-width of each inclination window [deg]
    min_samples : skip windows with fewer pixels than this

    Returns
    -------
    DataFrame with one SlidingGammaResult row per window
    """
    if gamma_centers is None:
        gamma_centers = np.arange(5, 86, 2)

    results: list[SlidingGammaResult] = []

    for g0 in gamma_centers:
        mask = (
                (gamma >= g0 - window_half_width)
                & (gamma <= g0 + window_half_width)
        )

        if np.sum(mask) < min_samples:
            continue

        y = target[mask]

        if np.unique(y).size < 2:
            continue

        bver_w = bver[mask]
        bhor_w = bhor[mask]

        X_combined = np.column_stack([bver_w, bhor_w])
        coef, intercept, auc_combined = fit_logistic_model(X_combined, y)

        _, _, auc_bver_only = fit_logistic_model(bver_w[:, None], y)
        _, _, auc_bhor_only = fit_logistic_model(bhor_w[:, None], y)

        results.append(
            SlidingGammaResult(
                gamma_center=g0,
                beta_ver=coef[0],
                beta_hor=coef[1],
                intercept=intercept,
                auc_combined=auc_combined,
                auc_bver_only=auc_bver_only,
                auc_bhor_only=auc_bhor_only,
                n_samples=int(np.sum(mask)),
                positive_fraction=float(np.mean(y)),
            )
        )

    return pd.DataFrame(results)
