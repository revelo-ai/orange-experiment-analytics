# pylint: disable=protected-access
import unittest
from unittest.mock import patch, Mock

import numpy as np
from AnyQt.QtCore import QItemSelection, QItemSelectionModel

from Orange.data import Table, Domain
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.experiment_analytics.excel_export import BorderRole
from orangecontrib.experiment_analytics.widgets.owletterreport import OWLetterReport, \
    ContextHandler
from orangewidget.settings import Context


class TestContextHandler(unittest.TestCase):
    def test_encode_setting(self):
        iris = Table("iris")
        context = Mock()
        setting = Mock()

        values = [[iris.domain.attributes[0], True, False],
                  [iris.domain.attributes[3], True, False]]
        encoded = ContextHandler.encode_setting(context, setting, values)
        res = ([[("sepal length", 102), True, False],
                [("petal width", 102), True, False]], -3)
        self.assertEqual(res, encoded)

        values = [iris.domain.attributes[0], iris.domain.attributes[3]]
        encoded = ContextHandler.encode_setting(context, setting, values)
        res = ([("sepal length", 102), ("petal width", 102)], -3)
        self.assertEqual(res, encoded)

    def test_decode_setting(self):
        iris = Table("iris")
        setting = Mock()

        value = ([[("sepal length", 102), True, False],
                  [("petal width", 102), True, False]], -3)
        decoded = ContextHandler().decode_setting(setting, value, iris.domain)
        res = [[iris.domain.attributes[0], True, False],
               [iris.domain.attributes[3], True, False]]
        self.assertEqual(res, decoded)

        value = ([("sepal length", 102), ("petal width", 102)], -3)
        decoded = ContextHandler().decode_setting(setting, value, iris.domain)
        res = [iris.domain.attributes[0], iris.domain.attributes[3]]
        self.assertEqual(res, decoded)

    def test_is_valid_item(self):
        handler = ContextHandler()
        setting = Mock()
        setting.exclude_attributes = False
        setting.exclude_metas = False
        attrs = {"sepal length": 2, "sepal width": 2,
                 "petal length": 2, "petal width": 2, "iris": 1}
        metas = {}

        item = ("iris", 101)
        self.assertTrue(handler.is_valid_item(setting, item, attrs, metas))

        item = [("iris", 101), True, False]
        self.assertTrue(handler.is_valid_item(setting, item, attrs, metas))

        item = [("foo", 101), True, False]
        self.assertFalse(handler.is_valid_item(setting, item, attrs, metas))


class TestOWLetterReport(WidgetTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.iris = Table("iris")

    def setUp(self):
        context = Context(
            attributes={"sepal length": 2, "sepal width": 2,
                        "petal length": 2, "petal width": 2, "iris": 1},
            metas={},
            values={"report_vars": ([[("sepal length", 102), True, False],
                                     [("sepal width", 102), True, False],
                                     [("petal length", 102), True, False],
                                     [("petal width", 102), True, False]], -3)}
        )
        settings = {"context_settings": [context]}
        self.widget = self.create_widget(OWLetterReport,
                                         stored_settings=settings)

    def test_input(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget._group_vars_model.rowCount(), 1)
        self.assertEqual(self.widget._report_vars_model.rowCount(), 4)

        table = [
            ["iris", "Iris-setosa", "Iris-versicolor", "Iris-virginica"],
            ["", "A", "B", "C"],
            ["Base Total Responses", "50", "50", "50"],
            ["sepal length - Mean", "5 (B,C)", "6 (A,C)", "7 (A,B)"],
            ["sepal width - Mean", "3 (B,C)", "3 (A,C)", "3 (A,B)"],
            ["petal length - Mean", "1 (B,C)", "4 (A,C)", "6 (A,B)"],
            ["petal width - Mean", "0 (B,C)", "1 (A,C)", "2 (A,B)"],
        ]
        model = self.widget._model
        for i in range(model.rowCount()):
            for j in range(model.columnCount()):
                self.assertEqual(table[i][j], model.data(model.index(i, j)))

    def test_input_error(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, self.iris[:3])
        self.assertEqual(self.widget._group_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._report_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._model.rowCount(), 0)
        self.assertTrue(self.widget.Error.not_enough_instances.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

        zoo = Table("zoo")
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, zoo)
        self.assertEqual(self.widget._group_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._report_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._model.rowCount(), 0)
        self.assertTrue(self.widget.Error.no_cont_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_cont_features.is_shown())

        housing = Table("housing")
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, housing)
        self.assertEqual(self.widget._group_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._report_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._model.rowCount(), 0)
        self.assertTrue(self.widget.Error.no_disc_features.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_disc_features.is_shown())

    @patch("orangecontrib.experiment_analytics.widgets.owletterreport.MAX_GROUPS", 2)
    def test_input_too_many_groups(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Error.too_many_groups.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.too_many_groups.is_shown())
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertTrue(self.widget.Error.too_many_groups.is_shown())
        self.send_signal(self.widget.Inputs.data, self.iris[:100])
        self.assertFalse(self.widget.Error.too_many_groups.is_shown())

    def test_input_missing_values(self):
        iris = self.iris.copy()
        iris.X[50, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, iris)
        self.assertEqual(self.widget._model.rowCount(), 7)
        self.assertFalse(self.widget.Error.not_enough_groups.is_shown())
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, iris[49:53])
        self.assertTrue(self.widget.Error.not_enough_groups.is_shown())
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, iris[48:52])
        self.assertFalse(self.widget.Error.not_enough_groups.is_shown())
        self.assertFalse(self.widget.Error.not_enough_instances.is_shown())

    def test_input_missing_group(self):
        self.send_signal(self.widget.Inputs.data, self.iris[50:])
        self.assertEqual(self.widget._model.rowCount(), 7)

    def test_input_missing_single_group(self):
        self.send_signal(self.widget.Inputs.data, self.iris[:50])
        self.assertEqual(self.widget._model.rowCount(), 0)
        self.assertTrue(self.widget.Error.not_enough_groups.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.not_enough_groups.is_shown())

    def test_input_missing_values_all(self):
        iris = self.iris.copy()
        iris.X[:50, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, iris)
        model = self.widget._report_vars_model
        model.setData(model.index(0, 2), True)
        table = [
            ["iris", "Iris-setosa", "Iris-versicolor", "Iris-virginica"],
            ["", "A", "B", "C"],
            ["Base Total Responses", "50", "50", "50"],
            ["sepal length - Mean", "?", "6 (B)", "7 (A)"],
            ["sepal length - % TB", "?", "2%", "2%"],
            ["sepal width - Mean", "3 (B,C)", "3 (A,C)", "3 (A,B)"],
            ["petal length - Mean", "1 (B,C)", "4 (A,C)", "6 (A,B)"],
            ["petal width - Mean", "0 (B,C)", "1 (A,C)", "2 (A,B)"],
        ]
        model = self.widget._model
        for i in range(model.rowCount()):
            for j in range(model.columnCount()):
                self.assertEqual(table[i][j], model.data(model.index(i, j)))

    def test_remove_input(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget._group_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._report_vars_model.rowCount(), 0)
        self.assertEqual(self.widget._model.rowCount(), 0)

    def test_multiple_group_by(self):
        heart = Table("heart_disease")
        self.send_signal(self.widget.Inputs.data, heart)

        sel_model = self.widget._group_vars_view.selectionModel()
        model = self.widget._group_vars_view.model()

        selection = QItemSelection()
        selection.select(model.index(2, 0), model.index(2, 0))
        selection.select(model.index(4, 0), model.index(4, 0))
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

        table = [
            ["chest pain", "asymptomatic", "asymptomatic", "asymptomatic",
             "atypical ang", "atypical ang", "non-anginal", "non-anginal",
             "non-anginal", "typical ang", "typical ang"],
            ["rest ECG", "normal", "left vent hypertrophy", "ST-T abnormal",
             "normal", "left vent hypertrophy", "normal",
             "left vent hypertrophy", "ST-T abnormal", "normal",
             "left vent hypertrophy"],
            ["", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            ["Base Total Responses", "62", "79", "3", "31", "19", "49", "36",
             "1", "9", "14"],
            ["age - Mean", "56", "56 (D)", "56", "50 (B)", "54", "52",
             "56", "76", "54", "57"],
            ["rest SBP - Mean", "130", "133", "141", "124 (I)", "135", "129",
             "132", "140", "137", "144 (D)"],
            ["cholesterol - Mean", "239", "258", "283", "239",
             "254", "235", "258", "197", "237", "237"],
            ["max HR - Mean", "142 (D,E,F)", "140 (D,E,F,G)", "129",
             "163 (A,B)", "162 (A,B)", "156 (A,B)", "155 (B)", "116", "157",
             "155"],
            ["ST by exercise - Mean", "1 (C,D)", "1 (D,E,F)",
             "3 (A,D,E,F,G,I)", "0 (A,B,C,H)", "0 (B,C,H)", "1 (B,C)", "1 (C)",
             "1", "2 (D,E)", "1 (C)"],
            ["major vessels colored - Mean", "1", "1 (D,F)", "1",
             "0 (B)", "0", "0 (B)", "1", "0", "1", "0"]
        ]
        model = self.widget._model
        for i in range(model.rowCount()):
            for j in range(model.columnCount()):
                self.assertEqual(table[i][j], model.data(model.index(i, j)))

    def test_on_report_variables_changed(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        model = self.widget._report_vars_model

        model.setData(model.index(0, 1), False)
        self.assertEqual(self.widget._model.rowCount(), 6)

        model.setData(model.index(0, 2), True)
        self.assertEqual(self.widget._model.rowCount(), 7)

    def test_report_variables_settings(self):
        widget = self.create_widget(OWLetterReport)
        self.send_signal(widget.Inputs.data, self.iris, widget=widget)
        report_vars = [[self.iris.domain.attributes[0], False, False],
                       [self.iris.domain.attributes[1], False, False],
                       [self.iris.domain.attributes[2], False, False],
                       [self.iris.domain.attributes[3], False, False]]
        self.assertEqual(widget.report_vars, report_vars)

    def test_saved_report_variables(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        model = self.widget._report_vars_model
        model.setData(model.index(0, 1), False)
        report_vars = [[self.iris.domain.attributes[0], False, False],
                       [self.iris.domain.attributes[1], True, False],
                       [self.iris.domain.attributes[2], True, False],
                       [self.iris.domain.attributes[3], True, False]]
        self.assertEqual(self.widget.report_vars, report_vars)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWLetterReport, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.iris, widget=widget)
        self.assertEqual(widget.report_vars, report_vars)
        self.assertEqual(widget._report_vars_model[:], report_vars)

    def test_saved_report_variables_drag_and_drop(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        report_vars = [[self.iris.domain.attributes[1], False, False],
                       [self.iris.domain.attributes[2], True, False],
                       [self.iris.domain.attributes[0], True, False],
                       [self.iris.domain.attributes[3], True, False]]
        self.widget.report_vars = report_vars

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWLetterReport, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.iris, widget=widget)
        self.assertEqual(widget.report_vars, report_vars)
        self.assertEqual(widget._report_vars_model[:], report_vars)

    def test_group_variables_open_context(self):
        data = Table("heart_disease")
        self.send_signal(self.widget.Inputs.data, data)
        sel_model = self.widget._group_vars_view.selectionModel()
        model = self.widget._group_vars_view.model()

        selection = QItemSelection()
        selection.select(model.index(1, 0), model.index(1, 0))
        selection.select(model.index(2, 0), model.index(2, 0))
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)
        self.assertEqual(["gender", "chest pain"],
                         [var.name for var in self.widget.group_vars])

        domain = data.domain
        domain = Domain(domain.attributes[:1] + domain.attributes[2:],
                        domain.class_vars)
        data = data.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(["diameter narrowing"],
                         [var.name for var in self.widget.group_vars])

    def test_report_variables_open_context(self):
        self.send_signal(self.widget.Inputs.data, self.iris)

        model = self.widget._report_vars_model
        model.setData(model.index(0, 1), False)

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.data, self.iris)

        report_vars = [[self.iris.domain.attributes[0], False, False],
                       [self.iris.domain.attributes[1], True, False],
                       [self.iris.domain.attributes[2], True, False],
                       [self.iris.domain.attributes[3], True, False]]
        self.assertEqual(self.widget.report_vars, report_vars)

    def test_saved_lines(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        row = 6
        self.widget._OWLetterReport__on_toggle_line(row, True)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWLetterReport, stored_settings=settings)
        self.send_signal(widget.Inputs.data, self.iris, widget=widget)
        self.assertTrue(widget._model.index(row, 0).data(BorderRole))

    def test_send_report(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
