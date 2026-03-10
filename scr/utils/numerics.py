import numpy as np
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
import warnings
from typing import Literal

from scr.utils.decorators import reduce_like

from scr.config.numerics import NUM_EPS


def is_constant(
        array: np.ndarray | list | float,
        constant: float | None = None,
        axis: int | None = None,
        atol: float = NUM_EPS
) -> bool:
    if atol < 0.:
        raise ValueError('"atol" must be a non-negative number')

    array = np.array(array, dtype=float)

    if np.ndim(array) == 0:
        array = array[np.newaxis]

    if constant is None:  # return True if the array is constant along the axis
        ddof = return_ddof(array, axis=axis)

        return np.std(array, axis=axis, ddof=ddof) < atol

    else:  # return True if the array is equal to "constant" along the axis
        return np.all(np.abs(array - constant) < atol, axis=axis)


def return_ddof(
        array: np.ndarray,
        axis: int | None = None
) -> int:
    return 1 if np.size(array, axis) > 1 else 0


def return_mean_std(
        array: np.ndarray,
        axis: int | None = None,
        ddof: int | None = None
) -> tuple[np.floating, np.floating]:
    mean_value = np.nanmean(array, axis=axis)
    if ddof is None:
        ddof = return_ddof(array, axis=axis)

    std_value = np.nanstd(array, axis=axis, ddof=ddof)

    return mean_value, std_value


def denoise_array(
        array: np.ndarray,
        sigma: float,
        x: np.ndarray | None = None,
        remove_mean: bool = False,
        sum_or_int: Literal["sum", "int"] = "sum"
) -> np.ndarray:
    if x is None:
        x = np.arange(0., np.shape(array)[-1])  # 0. to convert it to float

    equidistant_measure = np.var(np.diff(x))

    if equidistant_measure == 0.:  # equidistant step -> gaussian_filter1d is faster
        step = x[1] - x[0]
        correction = gaussian_filter1d(np.ones(len(x)), sigma=float(sigma / step), mode="constant")
        array_denoised = gaussian_filter1d(array, sigma=float(sigma / step), mode="constant")

        array_denoised = normalise_in_columns(array_denoised, norm_vector=correction)

    else:  # transmission application
        # Gaussian filters in columns
        gaussian = norm.pdf(np.reshape(x, (len(x), 1)), loc=x, scale=sigma)

        # need num_filters x num_wavelengths
        if np.ndim(gaussian) == 1:
            gaussian = np.reshape(gaussian, (1, -1))
        if np.ndim(gaussian) > 2:
            raise ValueError("Filter must be 1-D or 2-D array.")

        if sum_or_int == "sum":
            gaussian = normalise_in_columns(gaussian)
            array_denoised = array @ gaussian
        else:
            gaussian = normalise_in_columns(gaussian, trapezoid(y=gaussian, x=x))
            array_denoised = trapezoid(y=np.einsum("...j, kj -> ...kj", array, gaussian), x=x)

    if remove_mean:  # here I assume that the noise has a zero mean
        mn = np.mean(array_denoised - array, axis=-1, keepdims=True)
    else:
        mn = 0.

    return array_denoised - mn


def find_outliers1D(
        y: np.ndarray,
        x: np.ndarray | None = None,
        z_thresh: float = 2.5,
        max_iter: int = np.inf,
        num_eps: float = NUM_EPS
) -> np.ndarray:
    if x is None:
        x = np.arange(len(y))

    if len(np.unique(x)) != len(x):
        raise ValueError('"x" input must be unique.')

    inds = np.argsort(x)
    x_iterate, y_iterate = x[inds], y[inds]

    z_thresh = np.clip(z_thresh, a_min=num_eps, a_max=None)

    num_iter = 0
    while True:
        num_iter += 1
        deriv = np.diff(y_iterate) / np.diff(x_iterate)
        mu, sigma = return_mean_std(deriv)
        z_score = (deriv - mu) / sigma if sigma > 0. else np.array([0.])

        positive = np.where(np.logical_or(z_score > z_thresh, ~np.isfinite(z_score)))[0]
        negative = np.where(np.logical_or(-z_score > z_thresh, ~np.isfinite(z_score)))[0]

        # noise -> the points are next to each other (overlap if compensated for "diff" shift)
        outliers = stack((np.intersect1d(positive, negative + 1), np.intersect1d(negative, positive + 1)))

        if 0 in positive or 0 in negative:  # first index is outlier
            outliers = stack(([0], outliers))

        # last index is outlier
        if (len(z_score) - 1) in positive or (len(z_score) - 1) in negative:  # -1 to count "len" from 0
            outliers = stack((outliers, [len(x_iterate) - 1]))

        if np.size(outliers) == 0 or num_iter > max_iter:
            break

        x_iterate, y_iterate = np.delete(x_iterate, outliers), np.delete(y_iterate, outliers)

    return np.where(~np.isin(x, x_iterate))[0]


def normalise_array(
        array: np.ndarray,
        axis: int | None = None,
        norm_vector: np.ndarray | None = None,
        norm_constant: float = 1.,
        num_eps: float = NUM_EPS
) -> np.ndarray:
    if norm_vector is None:
        norm_vector = np.nansum(array, axis=axis, keepdims=True)

    # to force correct dimensions (e.g. when passing the output of interp1d)
    if np.ndim(norm_vector) != np.ndim(array) and np.ndim(norm_vector) > 0:
        norm_vector = np.expand_dims(norm_vector, axis=axis)

    if np.any(np.abs(norm_vector) < num_eps):
        warnings.warn("You normalise with (almost) zero values. Check the normalisation vector.")

    return array / norm_vector * norm_constant


def normalise_in_columns(
        array: np.ndarray,
        norm_vector: np.ndarray | None = None,
        norm_constant: float = 1.
) -> np.ndarray:
    return normalise_array(array, axis=0, norm_vector=norm_vector, norm_constant=norm_constant)


def normalise_in_rows(
        array: np.ndarray,
        norm_vector: np.ndarray | None = None,
        norm_constant: float = 1.
) -> np.ndarray:
    return normalise_array(array, axis=1, norm_vector=norm_vector, norm_constant=norm_constant)


def stack(
        arrays: tuple | list,
        axis: int | None = None,
        reduce: bool = False
) -> np.ndarray:
    """
    concatenate arrays along the specific axis

    if reduce=True, the "arrays" tuple is processed in this way
    arrays = (A, B, C, D)
    stack((stack((stack((A, B), axis=axis), C), axis=axis), D), axis=axis)
    This is potentially slower but allows for concatenating e.g.
    A.shape = (2, 4, 4)
    B.shape = (3, 4)
    C.shape = (4,)
    res = stack((C, B, A), axis=0, reduce=True)
    res.shape = (3, 4, 4)
    res[0] == stack((C, B), axis=0)
    res[1:] == A
    """

    @reduce_like
    def _stack(
            arrays: tuple | list,
            axis: int | None = None
    ) -> np.ndarray:
        ndim = np.array([np.ndim(array) for array in arrays])
        _check_dims(ndim, reduce)

        if np.all(ndim == 1):  # vector + vector + ...
            if axis is None:  # -> vector
                return np.concatenate(arrays, axis=axis)
            else:  # -> 2-D array
                return np.stack(arrays, axis=axis)

        elif np.var(ndim) != 0:  # N-D array + (N-1)-D array + ... -> N-D array
            max_dim = np.max(ndim)

            # longest array
            shape = np.array(np.shape(arrays[np.argmax(ndim)]))
            shape[axis] = -1

            # reshape is dangerous; you can potentially stack e.g. 10x1 with 2x5x2 along axis=0 that is confusing
            # possible dimension difference is one; omit the -1 shape. The rest should be equal.
            shapes = [np.array(np.shape(array)) for array in arrays if np.ndim(array) < max_dim]
            if not np.all([sh in shape[shape > 0] for sh in shapes]):
                raise ValueError("Arrays of these dimensions cannot be stacked.")

            arrays = [np.reshape(array, shape) if np.ndim(array) < max_dim else array for array in arrays]

            return np.concatenate(arrays, axis=axis)

        elif is_constant(ndim):  # N-D array + N-D array + ... -> N-D array or (N+1)-D array
            ndim = ndim[0]
            if axis < ndim:  # along existing dimensions
                return np.concatenate(arrays, axis=axis)
            else:  # along a new dimension
                return np.stack(arrays, axis=axis)

    def _check_dims(
            ndim: np.ndarray,
            reduce: bool = False
    ) -> None:
        error_msg = ("Maximum allowed difference in dimension of concatenated arrays is one. "
                     "If you want to stack along higher dimensions, use a combination of stack and np.reshape.")

        if np.max(ndim) - np.min(ndim) > 1:
            if reduce:
                raise ValueError(error_msg)
            else:
                raise ValueError(f'{error_msg}\nUse "reduce=True" to unlock more general (but slower) stacking.')

    # 0-D arrays to 1-D arrays (e.g. add a number to a vector)
    arrays = [np.reshape(array, (1,)) if np.ndim(array) == 0 else np.array(array) for array in arrays]
    arrays = tuple([array for array in arrays if np.size(array) > 0])
    if len(arrays) == 0:
        arrays = (np.array([], dtype=int),)  # enable to stack(np.array([]))

    if reduce:
        return _stack.reduce(arrays, axis)
    else:
        return _stack(arrays, axis)
