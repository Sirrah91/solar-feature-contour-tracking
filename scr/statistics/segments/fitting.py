import numpy as np
from pwlf import PiecewiseLinFit

from scr.statistics.segments.simple_pwlf import piecewise_linear_fit


def _poor_improvement_penalisation(
        poor_count: int,
        attenuation: float,
) -> float:
    return 1. + poor_count / attenuation


def fit_optimal_piecewise_linear_model(
        t: np.ndarray,
        y: np.ndarray,
        *,
        max_segments: int = 10,
        min_rel_improvement: float = 0.5,
        grace_attempts: int | None = None,
        use_aic: bool = False,
        use_bic: bool = False,
        normalize_y: bool = False,
        verbose: bool = True
) -> tuple[PiecewiseLinFit | None, dict]:
    """
    Fit piecewise linear models with increasing number of segments and choose the optimal one.
    (Improved with grace_attempts before early stopping)

    Parameters
    ----------
    t : np.ndarray
        1D array of time or x-values.
    y : np.ndarray
        1D array of y-values (e.g. total flux).
    max_segments : int
        Maximum number of segments to try.
    min_rel_improvement : float
        Minimum absolute SSR improvement required to add another segment.
    grace_attempts : int or None
        How many poor improvements to tolerate before stopping. If None, computed from `min_rel_improvement`.
    use_aic : bool
        If True, use Akaike Information Criterion to select optimal model.
    use_bic : bool
        If True, use Bayesian Information Criterion instead of AIC.
    normalize_y : bool
        If True, normalize y to [0, 1] for numerical stability.
    verbose : bool
        If True, print progress.

    Returns
    -------
    best_model : pwlf.PiecewiseLinFit
        Fitted model with optimal number of segments.
    results : dict
        Dictionary with all errors, aic, bic, and breakpoints per segment count.
    """
    if normalize_y:
        y = y / np.nanmax(np.abs(y))

    errors, aics, bics, breakpoints = [], [], [], []
    best_model, best_score = None, np.inf
    n = np.count_nonzero(np.isfinite(t) & np.isfinite(y))

    poor_improvement_count = 0
    baseline_ssr = None  # set only when poor improvement is first seen

    attenuation = 5.

    # poor_improvement_penalisation(max_grace_attempts) * min_rel_improvement <= 1
    max_grace_attempts = np.floor((1. / min_rel_improvement - 1.) * attenuation)

    if grace_attempts is None:
        grace_attempts = max_grace_attempts
    grace_attempts = int(min(grace_attempts, max_grace_attempts))

    max_segments = int(np.clip(np.ceil(len(t) / 2) - 1, a_min=1, a_max=max_segments))
    for i in range(1, max_segments + 1):
        try:
            model = piecewise_linear_fit(x=t, y=y, n_segments=i)
        except Exception as e:
            if verbose:
                print(f"  Failed to fit {i} segments: {e}")
            break

        brk = model.fit_breaks
        ssr = model.ssr

        k = i + 1  # number of breakpoints
        aic = n * np.log(ssr / n) + 2 * k
        bic = n * np.log(ssr / n) + k * np.log(n)

        errors.append(ssr)
        aics.append(aic)
        bics.append(bic)
        breakpoints.append(brk)

        if verbose:
            print(f"Segments: {i}, SSR: {ssr:.4g}, AIC: {aic:.3f}, BIC: {bic:.3f}")

        # Early stopping logic
        if i > 1 and not use_aic and not use_bic:
            if baseline_ssr is None:
                rel_improvement = np.abs((errors[i - 2] - errors[i - 1]) / errors[i - 2])
                if rel_improvement < min_rel_improvement:
                    baseline_ssr = errors[i - 2]
                    poor_improvement_count = 1
                else:
                    poor_improvement_count = 0
            else:
                rel_improvement = np.abs((baseline_ssr - errors[i - 1]) / baseline_ssr)
                if rel_improvement < min_rel_improvement * _poor_improvement_penalisation(poor_improvement_count, attenuation):
                    poor_improvement_count += 1
                else:
                    baseline_ssr = None  # reset if we observe a good improvement
                    poor_improvement_count = 0

            if poor_improvement_count >= grace_attempts:
                if verbose:
                    print(
                        f"  Stopping at {i} segments after {grace_attempts} poor improvements (ΔSSR: {rel_improvement:.3%})")
                break
            elif poor_improvement_count > 0:
                continue  # do not save score and model
        """
        if i > 1 and not use_aic and not use_bic:
            rel_improvement = np.abs((errors[i - 2] - errors[i - 1]) / errors[i - 2])
            if rel_improvement < min_rel_improvement:
                poor_improvement_count += 1
                if poor_improvement_count >= grace_attempts:
                    if verbose:
                        print(f"  Stopping at {i} segments due to repeated small improvement (last ΔSSR: {rel_improvement:.3%})")
                    break
                elif poor_improvement_count > 0:
                    continue  # do not save score and model
            else:
                poor_improvement_count = 0  # reset if improvement is acceptable
        """
        # Update best model based on selected criterion
        score = aic if use_aic else bic if use_bic else ssr
        if score < best_score:
            best_score = score
            best_model = model

    results = {
        "ssr": errors,
        "aic": aics,
        "bic": bics,
        "breakpoints": breakpoints
    }

    if verbose and best_model is not None:
        n_segments = len(best_model.fit_breaks) - 1
        print(f"Selected model uses {n_segments} segment{'s' if n_segments > 1 else ''}.\n")

    return best_model, results
