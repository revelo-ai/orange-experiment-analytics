"""
Create a letter report using the insert-and-absorb algorithm for solving CLD.
"""
from typing import List, Callable

import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def simple_letter_report(
        treatments: List[np.ndarray],
        threshold: float = 0.05,
) -> List[List[str]]:
    """
    Create a simple letter report.

    Parameters
    ----------
    treatments : list
        List of arrays.

    threshold : float
        Threshold for significant difference between two treatments.

    Returns
    -------
    letters: list
        A list of string (concatenated letters).
    """
    # compute p-values
    assert len(treatments) <= 26
    endog = np.hstack(treatments)
    groups = np.hstack([np.full(treatment.shape, chr(i))
                        for i, treatment in enumerate(treatments, 65)])
    res = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=threshold)

    # compute letters
    matrix = _into_matrix(res.pvalues, len(treatments))
    mask = matrix <= threshold
    arr = np.array([chr(65 + i) for i in range(len(treatments))])
    letters_matrix = np.tile(arr, (len(treatments), 1))
    return [",".join(row[m]) for m, row in zip(mask, letters_matrix)]


def _into_matrix(
        p_values: np.ndarray,
        n_treatments: int
) -> np.ndarray:
    matrix = np.zeros((n_treatments, n_treatments))
    indices = np.triu_indices(n_treatments, 1)
    matrix[indices] = p_values
    matrix = matrix + matrix.T
    matrix[np.diag_indices(n_treatments)] = 1
    return matrix


def letter_report(
        treatments: List[np.ndarray],
        threshold: float = 0.05
) -> List[List[str]]:
    """
    Create a letter report.

    Parameters
    ----------
    treatments : list
        List of arrays.

    threshold : float
        Threshold for significant difference between two treatments.

    Returns
    -------
    letters: list
        A list of lists of letters.
    """

    # sort treatments by mean
    indices = np.argsort([np.mean(t) for t in treatments])[::-1]
    treatments = [treatments[i] for i in indices]

    # compute p-values
    endog = np.hstack(treatments)
    groups = np.hstack([np.full(treatment.shape, chr(i))
                        for i, treatment in enumerate(treatments, 65)])
    res = pairwise_tukeyhsd(endog=endog, groups=groups, alpha=threshold)

    # compute letters
    matrix = _cld(res.pvalues, len(treatments), threshold)
    letters = _compute_letters(matrix)

    # unsort letters
    unsorted_letters = [["None"]] * len(letters)
    for i, letters_ in zip(indices, letters):
        unsorted_letters[i] = letters_

    return unsorted_letters


def _cld(
        p_values: np.ndarray,
        n_treatments: int,
        threshold: float = 0.05
) -> np.ndarray:
    """
    Create a compact letter display using the insert-and-absorb algorithm.
    Before obtaining p_values, the treatments should be sorted my mean.

    Parameters
    ----------
    p_values : np.ndarray of shape (n_treatments x n_treatments)
        An array with p-values.

    n_treatments : int
        Number of treatment.

    threshold : float, optional, default = 0.05
        Threshold for significant difference between two treatments.

    Returns
    -------
    matrix: np.ndarray of shape (n_treatments x n_letters)
        An array of 0 and 1.
    """
    assert n_treatments > 1
    assert len(p_values) > 0

    p_values_gen = (p for p in p_values)
    matrix = np.ones((n_treatments, 1))
    for i in range(n_treatments):
        for j in range(i + 1, n_treatments):
            if next(p_values_gen) < threshold:
                matrix = _insert(matrix, i, j)
                matrix = _absorb(matrix)

    return matrix


def _insert(matrix: np.ndarray, t1_index: int, t2_index: int) -> np.ndarray:
    matrix1 = matrix.copy()
    matrix1[t2_index, :] = 0
    matrix2 = matrix.copy()
    matrix2[t1_index, :] = 0
    return np.hstack((matrix1, matrix2))


def _absorb(matrix: np.ndarray) -> np.ndarray:
    for i in range(matrix.shape[1] - 1, 0, -1):
        msk = matrix.astype(bool)
        if any((all((msk[:, i] & msk[:, j]) == msk[:, i]) and i != j)
               for j in range(matrix.shape[1])):
            matrix = np.delete(matrix, i, axis=1)
    return matrix


def _compute_letters(matrix: np.ndarray) -> List[List[str]]:
    shape = matrix.shape
    report = np.tile(np.arange(shape[1]), (shape[0], 1)) + 65.0
    report[matrix == 0] = np.nan
    return _to_chr_lst(report)


def _to_chr_lst(report: np.ndarray) -> List[List[str]]:
    return [[_to_chr(ordinal) for ordinal in ordinals] for ordinals in report]


def _to_chr(number: float):
    return "" if np.isnan(number) else chr(int(number))
