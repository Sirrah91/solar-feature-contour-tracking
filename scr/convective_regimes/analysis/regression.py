import numpy as np
import pandas as pd

from scr.convective_regimes.core.models import RegressionResult


def analyse_regression_with_auc_loss(
        results: pd.DataFrame,
        *,
        loss_tolerance: float = 0.01,
) -> pd.DataFrame:
    """
    Augment sliding-window results with AUC-loss diagnostics and critical field tracks.

    Computes the relative information loss when each component is dropped,
    and derives the combined critical field vector (Bver, Bhor) at each window.

    Parameters
    ----------
    results : DataFrame from sliding_gamma_analysis
    loss_tolerance : relative AUC loss below which a component is negligible

    Returns
    -------
    Original DataFrame extended with: loss_ver, loss_hor, Bcrit_combined,
    Bver_from_combined, Bhor_from_combined, Bver_crit_domin, Bhor_crit_domin
    """
    results = results.copy()

    results["loss_ver"] = (
            (results["auc_combined"] - results["auc_bver_only"])
            / results["auc_combined"]
    )
    results["loss_hor"] = (
            (results["auc_combined"] - results["auc_bhor_only"])
            / results["auc_combined"]
    )

    g = np.deg2rad(results["gamma_center"])
    denom = (
            results["beta_ver"] * np.cos(g)
            + results["beta_hor"] * np.sin(g)
    ).replace(0, np.nan)

    results["Bcrit_combined"] = -results["intercept"] / denom
    results["Bver_from_combined"] = results["Bcrit_combined"] * np.cos(g)
    results["Bhor_from_combined"] = results["Bcrit_combined"] * np.sin(g)

    results["Bver_crit_domin"] = np.nan
    results["Bhor_crit_domin"] = np.nan

    bver_dominates = results["loss_hor"] <= loss_tolerance
    results.loc[bver_dominates, "Bver_crit_domin"] = (
        results.loc[bver_dominates, "Bver_from_combined"]
    )

    bhor_dominates = results["loss_ver"] <= loss_tolerance
    results.loc[bhor_dominates, "Bhor_crit_domin"] = (
        results.loc[bhor_dominates, "Bhor_from_combined"]
    )

    return results


def analyse_regression(
        results: pd.DataFrame,
        *,
        auc_noise_floor: float = 0.01,
) -> RegressionResult:
    """
    Run AUC-loss analysis and return a typed RegressionResult.

    This is the typed entry point consumed by core/transitions.py
    and core/baselines.py.
    """
    analysed = analyse_regression_with_auc_loss(results, loss_tolerance=auc_noise_floor)

    return RegressionResult(
        gamma_centers=analysed["gamma_center"].to_numpy(),
        bver=analysed["Bver_from_combined"].to_numpy(),
        bhor=analysed["Bhor_from_combined"].to_numpy(),
        loss_ver=analysed["loss_ver"].to_numpy(),
        loss_hor=analysed["loss_hor"].to_numpy(),
    )


def analyse_regression_results(
        results: pd.DataFrame,
        *,
        dominance_threshold: float = 0.8,
) -> pd.DataFrame:
    """
    Analyse sliding-window results using coefficient dominance fractions.

    Older approach, superseded by analyse_regression_with_auc_loss for most
    use cases. Retained for comparison.

    Parameters
    ----------
    results : DataFrame from sliding_gamma_analysis
    dominance_threshold : fractional dominance to classify a pure regime

    Returns
    -------
    Original DataFrame extended with dominance fractions and critical fields
    """
    results = results.copy()

    results["abs_beta_ver"] = np.abs(results["beta_ver"])
    results["abs_beta_hor"] = np.abs(results["beta_hor"])

    coef_sum = (
            results["abs_beta_ver"] + results["abs_beta_hor"]
    ).replace(0, np.nan)

    results["ver_fraction"] = results["abs_beta_ver"] / coef_sum
    results["hor_fraction"] = results["abs_beta_hor"] / coef_sum

    results["dominant_regime"] = "mixed"
    results.loc[
        results["ver_fraction"] >= dominance_threshold, "dominant_regime"
    ] = "vertical"
    results.loc[
        results["hor_fraction"] >= dominance_threshold, "dominant_regime"
    ] = "horizontal"

    results["Bver_crit"] = np.nan
    results["Bhor_crit"] = np.nan

    mask_ver = results["dominant_regime"] == "vertical"
    results.loc[mask_ver, "Bver_crit"] = (
            -results.loc[mask_ver, "intercept"]
            / results.loc[mask_ver, "beta_ver"]
    )

    mask_hor = results["dominant_regime"] == "horizontal"
    results.loc[mask_hor, "Bhor_crit"] = (
            -results.loc[mask_hor, "intercept"]
            / results.loc[mask_hor, "beta_hor"]
    )

    g = np.deg2rad(results["gamma_center"])
    denom = (
            results["beta_ver"] * np.cos(g)
            + results["beta_hor"] * np.sin(g)
    ).replace(0, np.nan)

    results["Bcrit_combined"] = -results["intercept"] / denom
    results["Bver_from_combined"] = results["Bcrit_combined"] * np.cos(g)
    results["Bhor_from_combined"] = results["Bcrit_combined"] * np.sin(g)

    auc_sum = (
            results["auc_bver_only"] + results["auc_bhor_only"]
    ).replace(0, np.nan)
    results["auc_ver_fraction"] = results["auc_bver_only"] / auc_sum
    results["auc_hor_fraction"] = results["auc_bhor_only"] / auc_sum

    return results
