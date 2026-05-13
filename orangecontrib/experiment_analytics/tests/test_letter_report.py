# pylint: disable=missing-docstring, no-self-use
import unittest

import numpy as np

from Orange.data import Table
from orangecontrib.experiment_analytics.letter_report import _cld, _absorb, \
    _compute_letters, letter_report, simple_letter_report, _into_matrix


class TestSimpleLetterReport(unittest.TestCase):
    def test_simple_letter_report(self):
        iris = Table("iris")
        treatments = [iris.X[:50, 0], iris.X[50: 100, 0], iris.X[100:, 0]]
        self.assertEqual(simple_letter_report(treatments),
                         ["B,C", "A,C", "A,B"])

        heart = Table("heart_disease")
        treatments = [heart.X[heart.X[:, 2] == 0, 0],
                      heart.X[heart.X[:, 2] == 1, 0],
                      heart.X[heart.X[:, 2] == 2, 0],
                      heart.X[heart.X[:, 2] == 3, 0]]
        self.assertEqual(simple_letter_report(treatments),
                         ["B", "A", "", ""])

    def test_into_matrix(self):
        n_treatments = 5
        arr = np.arange(n_treatments * (n_treatments - 1) / 2)
        res = np.array([[1, 0, 1, 2, 3],
                        [0, 1, 4, 5, 6],
                        [1, 4, 1, 7, 8],
                        [2, 5, 7, 1, 9],
                        [3, 6, 8, 9, 1]])
        np.testing.assert_array_equal(_into_matrix(arr, n_treatments), res)


class TestLetterReport(unittest.TestCase):
    def test_letter_report(self):
        iris = Table("iris")
        treatments = [iris.X[:50, 0], iris.X[50: 100, 0], iris.X[100:, 0]]
        self.assertEqual(
            letter_report(treatments),
            [["", "", "C"],
             ["", "B", ""],
             ["A", "", ""]]
        )

        heart = Table("heart_disease")
        treatments = [heart.X[heart.X[:, 2] == 0, 0],
                      heart.X[heart.X[:, 2] == 1, 0],
                      heart.X[heart.X[:, 2] == 2, 0],
                      heart.X[heart.X[:, 2] == 3, 0]]
        self.assertEqual(
            letter_report(treatments),
            [["A", ""],
             ["", "B"],
             ["A", "B"],
             ["A", "B"]])

    def test_cld_type(self):
        p_values = np.array([1, 1, 1])
        self.assertIsInstance(_cld(p_values, 3), np.ndarray)

    def test_cld_size(self):
        p_values = np.array([1, 1, 1])
        self.assertEqual(len(_cld(p_values, 3)), 3)

    def test_cld_values(self):
        p_values = np.array([0.04])
        self.assertEqual(list(np.unique(_cld(p_values, 2))), [0, 1])

    def test_cld_sig_same(self):
        p_values = np.array([1, 1, 1])
        np.testing.assert_array_equal(_cld(p_values, 3), np.ones((3, 1)))

    def test_cld_sig_diff(self):
        p_values = np.array([0.9, 0.7, 0.1, 0.02, 0.01, 0.8, 0.1, 0.03,
                             0.01, 0.5, 0.3, 0.02, 0.4, 0.05, 0.04])
        res = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0],
                        [0, 0, 1]])
        np.testing.assert_array_equal(_cld(p_values, 6), res)

    def test_absorb(self):
        matrix = np.ones((4, 2))
        np.testing.assert_array_equal(_absorb(matrix), np.ones((4, 1)))

        matrix[0, 0] = 0
        matrix[3, 1] = 0
        np.testing.assert_array_equal(_absorb(matrix), matrix)

        matrix = np.array([[0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [1, 0, 1, 0],
                           [1, 0, 0, 0]])
        res = np.array([[0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [1, 0, 0]])
        np.testing.assert_array_equal(_absorb(matrix), res)

    def test_compute_letters(self):
        p_values = np.array([0.9, 0.7, 0.1, 0.02, 0.01, 0.8, 0.1, 0.03,
                             0.01, 0.5, 0.3, 0.02, 0.4, 0.05, 0.04])
        res = [["A", "", ""],
               ["A", "", ""],
               ["A", "B", ""],
               ["A", "B", "C"],
               ["", "B", ""],
               ["", "", "C"]]
        report = _compute_letters(_cld(p_values, 6))
        self.assertEqual(report, res)


if __name__ == "__main__":
    unittest.main()
