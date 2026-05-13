import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from orangecontrib.experiment_analytics.aggregate.frequency import frequency


class TestFrequency(TestCase):
    def setUp(self) -> None:
        x_ = np.linspace(0.1, 20 * np.pi, 1000)
        # period = 2pi => 0.159 Hz
        self.data = pd.DataFrame({"x": x_, "y": np.sin(x_)})

    def test_basics(self):
        res = frequency(self.data, False, False)
        expected = pd.DataFrame(
            {("y", "Frequency"): [0.159], ("y", "Amplitude"): [0.99]}
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        res = frequency(self.data, False, True)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [0.159],
                ("y", "Amplitude"): [0.99],
                ("y", "R2"): [1.0],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        res = frequency(self.data,True, False)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [0.159],
                ("y", "Amplitude"): [0.99],
                ("y", "Half-life"): [-1023824.13],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        res = frequency(self.data, True, True)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [0.159],
                ("y", "Amplitude"): [0.99],
                ("y", "Half-life"): [-1023824.13],
                ("y", "R2"): [1.0],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

    def test_sampling(self):
        x_ = np.linspace(0.1, 20 * np.pi, 1000000)
        data = pd.DataFrame({"x": x_, "y": np.sin(x_)})

        res = frequency(data, False, False)
        expected = pd.DataFrame(
            {("y", "Frequency"): [0.159], ("y", "Amplitude"): [0.99]}
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

    def test_missing_values(self):
        self.data.iloc[0, 0] = np.nan
        res = frequency(self.data, False, False)
        expected = pd.DataFrame(
            {("y", "Frequency"): [0.159], ("y", "Amplitude"): [0.99]}
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        self.data.iloc[0, 1] = np.nan
        res = frequency(self.data, False, False)
        expected = pd.DataFrame(
            {("y", "Frequency"): [0.159], ("y", "Amplitude"): [0.99]}
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

    def test_not_enough_values(self):
        res = frequency(self.data.iloc[:1], True, True)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [np.nan],
                ("y", "Amplitude"): [np.nan],
                ("y", "Half-life"): [np.nan],
                ("y", "R2"): [np.nan],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        self.data.iloc[1, 0] = 0.1  # make x of second row same than x in first
        res = frequency(self.data.iloc[:2], True, True)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [np.nan],
                ("y", "Amplitude"): [np.nan],
                ("y", "Half-life"): [np.nan],
                ("y", "R2"): [np.nan],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

    def test_de_trending(self):
        x = np.linspace(0.1, 20 * np.pi, 1000)
        y = np.sin(x) + x * 0.2
        data = pd.DataFrame({"x": x, "y": y})
        res = frequency(data, False, False)
        expected = pd.DataFrame(
            {("y", "Frequency"): [0.159], ("y", "Amplitude"): [0.99]}
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        x = np.linspace(0.1, 20 * np.pi, 1000)
        y = np.sin(x) + x**2 * 0.2
        data = pd.DataFrame({"x": x, "y": y})
        res = frequency(data, False, False, detrend_degree=2)
        expected = pd.DataFrame(
            {("y", "Frequency"): [0.159], ("y", "Amplitude"): [0.99]}
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        res = frequency(data, False, False, detrend_degree=None)
        expected = pd.DataFrame(
            {("y", "Frequency"): [0.003], ("y", "Amplitude"): [488.73]}
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

    def test_damping(self):
        x = np.linspace(0.1, 20 * np.pi, 1000)
        y = np.sin(x) / ((x + 1) * 0.005)
        data = pd.DataFrame({"x": x, "y": y})
        res = frequency(data, True, False)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [0.159],
                ("y", "Amplitude"): [85.25],
                ("y", "Half-life"): [4.79],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

    def test_fitting_failed(self):
        x = np.linspace(0.1, 20 * np.pi, 1000)
        y = 1 / (x**3)
        data = pd.DataFrame({"x": x, "y": y})
        res = frequency(data, True, True)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [0.023],
                ("y", "Amplitude"): [3.95],
                ("y", "Half-life"): [np.nan],
                ("y", "R2"): [np.nan],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)

        res = frequency(data, True, False)
        expected = pd.DataFrame(
            {
                ("y", "Frequency"): [0.023],
                ("y", "Amplitude"): [3.95],
                ("y", "Half-life"): [np.nan],
            }
        )
        pd.testing.assert_frame_equal(res, expected, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
