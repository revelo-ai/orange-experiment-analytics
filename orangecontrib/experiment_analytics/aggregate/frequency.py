from functools import partial
from typing import Callable, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import lombscargle
from sklearn.metrics import r2_score

COLUMNS = np.array(["Frequency", "Amplitude", "Half-life", "R2"])


def frequency(
    df: pd.DataFrame, use_damping: bool, compute_r2: bool, detrend_degree: Optional[int] = 1
) -> pd.DataFrame:
    """
    The frequency aggregation function, that computes the dominant frequency
    and its amplitude.

    When use_damping is True it fits a damped cosine model with exponential
    decay using least squares optimization to find the frequency, amplitude and
    half-life (damping rate).

    If compute_r2 is True, fit a model with the inferred parameters and report
    the R2 of its predictions.

    Parameters
    ----------
    use_damping
        The signal has damping -- fit model that corrects the frequency and
        amplitude and report the damping rate
    compute_r2
        Compute R2 of the single-frequency model predictions
    df
        The data -- first column is time and the second function values
    detrend_degree
        The degree of polynomial function used for the de-trending of the signal.
        Use None to disable de-trending.

    Returns
    -------
    The one line DataFrame that reports the frequency, amplitude, damping rate
    and r2 score.
    """
    # sample data to maximally 1000 points -- Lomb-Scargle method complexity
    # depends on number of points
    df = df.dropna()
    df = df.sample(n=min(len(df), 1000), axis=0, random_state=0)
    df = df.sort_values(df.columns[0])
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    columns = __column_names(use_damping, compute_r2, y.name)

    if len(x.unique()) < 2:
        # cannot fit and define the grid when less than 2 unique x values
        return pd.DataFrame([[np.nan] * len(columns)], columns=columns)

    x_det, y_det = __de_trending(x, y, detrend_degree)
    f, a = __periodogram(x_det, y_det)

    res = [f, a]
    if use_damping or compute_r2:
        # when damping in signal use model to correct frequency, amplitude and
        # compute the damping rate
        # when compute_r2 model reconstructs the signal to compute r2
        try:
            popt, fun = __fit_model(x_det, y_det, f, a, use_damping)
            if use_damping:
                f, a, decay = popt[1:4]
                res = [f, a, np.log(2) / decay]
            if compute_r2:
                y_recon = fun(x_det, *popt)
                compute_r2 = r2_score(y_det, y_recon)
                res.append(compute_r2)
        except RuntimeError:
            # fitting model may fail for some functions, it will usually not
            # fail for sinus like signal - report just frequency and amplitude
            # by the Lomb-Scargle
            res += [np.nan for _ in columns[2:]]

    return pd.DataFrame([res], columns=columns)


def __column_names(damping: bool, r2: bool, series_name: str) -> pd.MultiIndex:
    columns = COLUMNS[[True, True, damping, r2]]
    return pd.MultiIndex.from_tuples([(series_name, c) for c in columns])


def __freq_grid(x: pd.Series) -> np.ndarray:
    """
    Compute the frequency grid that is used by Lomb-Scargle to search for
    dominating frequency. Intuition behind can be found in
    https://jakevdp.github.io/blog/2015/06/13/lomb-scargle-in-python/#Frequency-spacing
    or in paper Understanding the Lomb–Scargle Periodogram by VanderPlas JT
    """
    n = len(x)
    span = x.max() - x.min()
    diff = 1 / (5 * span)
    return np.arange(diff, (n - 1) / 2 / span, diff)


def __de_trending(x: pd.Series, y: pd.Series, deg: Optional[int]) -> Tuple[pd.Series, pd.Series]:
    """
    De-trend the signal with a polynomial function with degree deg:
    1. Fit a polynomial function to the signal
    2. Subtract the value of the polynomial from the signal
    `x` is just shifted so that the minimum value is 0
    """
    if deg is None:
        return x, y
    p = np.polyfit(x, y, deg=deg)
    trend = np.polyval(p, x)
    y_det = y - trend
    x_det = x - x.min()
    return x_det, y_det


def __periodogram(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    """
    Use Lomb-Scargle method to compute the periodogram and extract the
    dominating frequency and its amplitude.
    """
    freqs = __freq_grid(x)
    pgram = lombscargle(x, y, freqs * 2 * np.pi, normalize=False)
    ind = np.argmax(pgram)
    amplitude = np.sqrt(pgram[ind] / len(x) * 4.0)
    return freqs[ind], amplitude


def __periodic_f_decay(x, theta, f, a, decay) -> float:
    """Function used to model the data to estimate damping and/or r2"""
    ret = a * np.cos(f * 2 * np.pi * x + theta)
    if decay is not None:
        ret *= np.exp(-decay * x)
    return ret


def __fit_model(
    x: pd.Series, y: pd.Series, f: float, a: float, use_damping: bool
) -> Tuple[np.ndarray, Callable]:
    """
    Fit the periodic model using non-linear least squares:
    - when `use_damping` is `True`, fit frequency, amplitude and damping rate in addition
      to the phase, starting with the frequency and amplitude retrieved by Lomb-Scargle
    - otherwise, only fit the phase, using a fixed frequency and amplitude from
      Lomb-Scargle (to measure r2 and estimate the goodness of fit)
    """
    if use_damping:
        fun = __periodic_f_decay
        p0 = (np.pi, f, a, 0)
        # frequency and amplitude should be positive values
        bounds = ([-np.inf, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf])
    else:
        fun = partial(__periodic_f_decay, a=a, f=f, decay=None)
        p0 = (np.pi,)
        bounds = (-np.inf, np.inf)
    return curve_fit(fun, x, y, p0=p0, bounds=bounds)[0], fun


if __name__ == "__main__":
    x_ = np.linspace(0.1, 20 * np.pi, 1000)
    y_ = np.sin(x_) / (x_ * 0.2)
    df_ = pd.DataFrame({"x": x_, "y": y_})
    print(frequency(False, False, df_))
    print(frequency(True, False, df_))
    print(frequency(False, True, df_))
    print(frequency(True, True, df_))
