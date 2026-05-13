# pylint: disable=protected-access
import unittest

import numpy as np

from AnyQt.QtCore import Qt
from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.experiment_analytics.widgets.owcomparemeans import OWCompareMeans


class TestOWCompareMeans(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")

    def setUp(self):
        self.widget = self.create_widget(OWCompareMeans)

    def test_input(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget._value_var_model.rowCount(), 4)
        self.assertEqual(self.widget._group_var_model.rowCount(), 1)
        self.assertEqual(self.widget._model.rowCount(), 3)

        table = [
            ["Iris-virginica", "A", "", "", "50", "6.588", "0.629", "0.096"],
            ["Iris-versicolor", "", "B", "", "50", "5.936", "0.511", "0.086"],
            ["Iris-setosa", "", "", "C", "50", "5.006", "0.349", "0.070"],
        ]
        model = self.widget._model
        for i in range(model.rowCount()):
            for j in range(model.columnCount()):
                self.assertEqual(table[i][j], model.data(model.index(i, j)))

    def test_input_error(self):
        self.send_signal(self.widget.Inputs.data, self.iris[:3])
        self.assertEqual(self.widget._value_var_model.rowCount(), 0)
        self.assertEqual(self.widget._group_var_model.rowCount(), 0)
        self.assertTrue(self.widget.Error.not_enough_instances.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

        zoo = Table("zoo")
        self.send_signal(self.widget.Inputs.data, zoo)
        self.assertEqual(self.widget._value_var_model.rowCount(), 0)
        self.assertEqual(self.widget._group_var_model.rowCount(), 0)
        self.assertTrue(self.widget.Error.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_cont_features.is_shown())

        housing = Table("housing")
        self.send_signal(self.widget.Inputs.data, housing)
        self.assertEqual(self.widget._value_var_model.rowCount(), 0)
        self.assertEqual(self.widget._group_var_model.rowCount(), 0)
        self.assertTrue(self.widget.Error.no_disc_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_disc_features.is_shown())

    def test_input_missing_values(self):
        iris = self.iris.copy()
        iris.X[50, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, iris)
        self.assertEqual(self.widget._model.rowCount(), 3)
        self.assertFalse(self.widget.Error.not_enough_treatments.is_shown())
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, iris[49:53])
        self.assertTrue(self.widget.Error.not_enough_treatments.is_shown())
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, iris[48:52])
        self.assertTrue(self.widget.Error.not_enough_treatments.is_shown())
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, iris[48:53])
        self.assertFalse(self.widget.Error.not_enough_treatments.is_shown())
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

    def test_input_missing_group(self):
        self.send_signal(self.widget.Inputs.data, self.iris[50:])
        self.assertEqual(self.widget._model.rowCount(), 2)

    def test_input_missing_single_group(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget._model.rowCount(), 3)

        self.send_signal(self.widget.Inputs.data, self.iris[:50])
        self.assertEqual(self.widget._model.rowCount(), 0)
        self.assertTrue(self.widget.Error.not_enough_treatments.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_treatments.is_shown())

    def test_remove_input(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget._model.rowCount(), 3)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget._model.rowCount(), 0)

    def test_view_header_labels(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget._view.model().headerData(0, Qt.Horizontal), "iris")

    def test_main_area_info(self):
        self.assertEqual(self.widget.variable_name, "")
        self.assertEqual(self.widget.result_anova, "")

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.variable_name, "sepal length")
        self.assertEqual(self.widget.result_anova, "119.265 (p=0.000)")

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.variable_name, "")
        self.assertEqual(self.widget.result_anova, "")

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
