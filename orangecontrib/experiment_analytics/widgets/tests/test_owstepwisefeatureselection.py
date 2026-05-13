import time
import unittest
from operator import attrgetter
from typing import Iterable
from unittest.mock import patch, ANY

import numpy as np
from AnyQt.QtCore import Qt
from AnyQt.QtTest import QTest
from AnyQt.QtWidgets import QSpinBox, QDoubleSpinBox

from Orange.classification import TreeLearner, LogisticRegressionLearner
from Orange.classification.logistic_regression import LogisticRegressionClassifier
from Orange.classification.majority import ConstantModel
from Orange.data import Table, Domain, ContinuousVariable
from Orange.modelling import RandomForestLearner
from Orange.regression import LinearRegressionLearner
from Orange.regression.linear import LinearModel
from Orange.regression.mean import MeanModel
from Orange.tree import TreeModel
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    InfoTransform,
    HTML_TABLE_STYLE,
)
from orangecontrib.experiment_analytics.widgets.owstepwisefeatureselection import (
    OWStepwiseFeatureSelection,
    ColoredBarItemDelegate,
    StepwiseFeatureSelectionTransform,
)


WIDGET_PATH = (
    "orangecontrib.experiment_analytics.widgets.owstepwisefeatureselection.OWStepwiseFeatureSelection"
)


class TestColoredBarItemDelegate(unittest.TestCase):
    def test_display_text(self):
        delegate = ColoredBarItemDelegate(decimals=4)
        # test positive decimal numbers
        self.assertEqual("10.1000", delegate.displayText(10.1))
        self.assertEqual("1.0100", delegate.displayText(1.01))
        self.assertEqual("0.1010", delegate.displayText(0.101))
        self.assertEqual("0.0101", delegate.displayText(0.0101))
        self.assertEqual("0.0010", delegate.displayText(0.00101))
        self.assertEqual("0.0001", delegate.displayText(0.000101))
        self.assertEqual("1.010e-05", delegate.displayText(0.0000101))
        self.assertEqual("1.010e-06", delegate.displayText(0.00000101))
        # test negative decimal numbers
        self.assertEqual("-10.1000", delegate.displayText(-10.1))
        self.assertEqual("-1.0100", delegate.displayText(-1.01))
        self.assertEqual("-0.1010", delegate.displayText(-0.101))
        self.assertEqual("-0.0101", delegate.displayText(-0.0101))
        self.assertEqual("-0.0010", delegate.displayText(-0.00101))
        self.assertEqual("-0.0001", delegate.displayText(-0.000101))
        self.assertEqual("-1.010e-05", delegate.displayText(-0.0000101))
        self.assertEqual("-1.010e-06", delegate.displayText(-0.00000101))


class TestOWStepwiseFeatureSelection(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWStepwiseFeatureSelection)
        self.iris = Table("iris")
        self.housing = Table("housing")

    def set_checked(self, row, col, checked):
        model = self.widget.features_model
        model.setData(model.index(row, col), checked, Qt.CheckStateRole)
        self.wait_until_finished()

    def set_locked(self, index, locked):
        self.set_checked(index, 0, locked)

    def set_selected(self, index, locked):
        self.set_checked(index, 1, locked)

    @staticmethod
    def sorted_vars(variables: Iterable):
        return sorted([a.name for a in variables])

    def assert_checked(self, expected, col):
        model = self.widget.features_model
        cs = Qt.CheckStateRole
        expected = [Qt.Checked if x else Qt.Unchecked for x in expected]
        state = [model.data(model.index(i, col), cs) for i in range(model.rowCount())]
        self.assertListEqual(expected, state)

    def assert_locked(self, expected):
        return self.assert_checked(expected, 0)

    def assert_selected(self, expected):
        return self.assert_checked(expected, 1)

    def scores(self):
        model = self.widget.features_table.model()
        return [model.data(model.index(i, 3)) for i in range(model.rowCount())]

    def stats(self):
        model = self.widget.scores_model
        n = model.columnCount()
        return [float(model.data(model.index(0, i))) for i in range(n)]

    def assert_list_almost_equal(self, l1, l2):
        """Compare two list that contain both numbers and strings"""
        self.assertEqual(len(l1), len(l2))
        for v1, v2 in zip(l1, l2):
            if isinstance(v1, str):
                self.assertEqual(v1, v2)
            else:
                self.assertAlmostEqual(v1, v2, delta=0.01)

    def test_data(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.iris, self.widget.data)
        self.assertEqual(len(self.iris.domain.attributes), self.widget.num_all_attr)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.data)
        self.assertEqual(0, self.widget.num_all_attr)

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertEqual(self.housing, self.widget.data)
        self.assertEqual(len(self.housing.domain.attributes), self.widget.num_all_attr)

        # try data without class
        data = self.iris.transform(Domain(self.iris.domain.attributes))
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.class_required.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(0, self.widget.features_table.model().rowCount())
        self.assertEqual(0, self.widget.scores_model.rowCount())
        self.assertListEqual([], self.widget.entered_features)
        self.assertListEqual([], self.widget.locked_features)
        self.assertEqual(0, self.widget.num_all_attr)

    def test_learner(self):
        outputs = self.widget.Outputs

        learner = TreeLearner()
        self.send_signal(self.widget.Inputs.learner, learner)
        self.assertEqual(learner, self.widget.learner)
        self.wait_until_finished()
        self.assertEqual(learner, self.widget.stepwise_fs.scorer.learner)
        self.assertIsNone(self.get_output(outputs.model))

        pp = self.get_output(outputs.preprocessor)
        self.assertEqual(learner, pp.sfs.scorer.learner)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(learner, self.widget.learner)
        self.wait_until_finished()
        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assertEqual(learner, self.widget.stepwise_fs.scorer.learner)
        self.assertIsInstance(self.get_output(outputs.model), TreeModel)

        pp = self.get_output(outputs.preprocessor)
        self.assertEqual(learner, pp.sfs.scorer.learner)

        self.send_signal(self.widget.Inputs.learner, None)
        self.assertIsNone(self.widget.learner)
        self.wait_until_finished()
        self.assertIsNone(self.widget.stepwise_fs.scorer.learner)

        self.assertIsNone(self.get_output(outputs.preprocessor).sfs.scorer.learner)
        model = self.get_output(outputs.model)
        self.assertIsInstance(model, LogisticRegressionClassifier)

    def test_context(self):
        w = self.widget
        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()

        w.step_btn.click()
        self.wait_until_finished()
        self.assertListEqual(["petal length"], self.sorted_vars(w.entered_features))
        w.step_btn.click()
        self.set_locked(1, True)

        exp = ["petal length", "petal width"]
        self.assertListEqual(exp, self.sorted_vars(w.entered_features))
        self.assertListEqual(["sepal width"],
                             self.sorted_vars(w.locked_features))

        self.send_signal(w.Inputs.data, self.housing)
        self.assertListEqual([], w.entered_features)
        self.assertListEqual([], w.locked_features)

        self.send_signal(w.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertListEqual([], w.entered_features)
        self.assertListEqual([], w.locked_features)
        self.assert_locked([False] * 4)
        self.assert_selected([False] * 4)

    def test_scores_table(self):
        model = self.widget.scores_model

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertEqual(5, self.widget.scores_model.columnCount())
        self.assertEqual(1, self.widget.scores_model.rowCount())
        self.assertListEqual(
            ["AUC", "CA", "F1", "Prec", "Recall"],
            [model.headerData(c, Qt.Horizontal) for c in range(5)],
        )
        self.assert_list_almost_equal([0.5, 0.333, 0.167, 0.111, 0.333], self.stats())

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assertEqual(5, self.widget.scores_model.columnCount())
        self.assertEqual(1, self.widget.scores_model.rowCount())
        self.assertListEqual(
            ["AUC", "CA", "F1", "Prec", "Recall"],
            [model.headerData(c, Qt.Horizontal) for c in range(5)],
        )
        self.assert_list_almost_equal([0.99, 0.94, 0.94, 0.94, 0.94], self.stats())

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.wait_until_finished()
        self.assertEqual(3, self.widget.scores_model.columnCount())
        self.assertEqual(1, self.widget.scores_model.rowCount())
        self.assertListEqual(
            ["R2", "RMSE", "MAE"],
            [model.headerData(c, Qt.Horizontal) for c in range(3)],
        )
        self.assert_list_almost_equal([-0.00, 9.20, 6.66], self.stats())

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assertEqual(3, self.widget.scores_model.columnCount())
        self.assertEqual(1, self.widget.scores_model.rowCount())
        self.assertListEqual(
            ["R2", "RMSE", "MAE"],
            [model.headerData(c, Qt.Horizontal) for c in range(3)],
        )
        self.assert_list_almost_equal([0.54, 6.22, 4.52], self.stats())

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()
        self.assertEqual(0, self.widget.scores_model.columnCount())

    def assert_bold(self, expected):
        m = self.widget.features_model
        for col in range(4):
            font = [m.data(m.index(i, col), Qt.FontRole) for i in range(m.rowCount())]
            self.assertListEqual(expected, [False if not f else f.bold() for f in font])

    def test_features_table(self):
        model = self.widget.features_model

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertEqual(4, model.columnCount())
        n = len(self.iris.domain.attributes)
        self.assertEqual(n, model.rowCount())
        self.assertListEqual(
            ["Locked", "Entered", "Feature", "Score difference"],
            [model.headerData(c, Qt.Horizontal).strip() for c in range(4)],
        )
        # check nothing in checked by default in first two columns
        self.assert_locked([False] * n)
        self.assert_selected([False] * n)

        # check attributes
        self.assertEqual(
            list(map(attrgetter("name"), self.iris.domain.attributes)),
            [model.data(model.index(i, 2)) for i in range(n)],
        )
        self.assert_list_almost_equal([0.367, 0.261, 0.492, 0.491], self.scores())
        self.assert_bold([False, False, True, False])

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.wait_until_finished()
        self.assertEqual(4, model.columnCount())
        n = len(self.housing.domain.attributes)
        self.assertEqual(n, model.rowCount())
        self.assertListEqual(
            ["Locked", "Entered", "Feature", "Score difference"],
            [model.headerData(c, Qt.Horizontal).strip() for c in range(4)],
        )
        # check nothing in checked by default in first two columns
        self.assert_locked([False] * n)
        self.assert_selected([False] * n)

        # check attributes
        self.assertEqual(
            list(map(attrgetter("name"), self.housing.domain.attributes)),
            [model.data(model.index(i, 2)) for i in range(n)],
        )
        ex = [
            0.13, 0.12, 0.23, 0.02, 0.18, 0.48, 0.14, 0.06, 0.14, 0.21, 0.25, 0.11, 0.54
        ]
        self.assert_list_almost_equal(ex, self.scores())
        self.assert_bold([False] * 12 + [True])

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()
        self.assertEqual(4, model.columnCount())
        self.assertEqual(0, model.rowCount())

    def test_score_changed(self):
        model = self.widget.features_model
        n = len(self.iris.domain.attributes)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        # default score is AUC
        self.assertEqual("AUC", self.widget.stepwise_fs.scorer.method)
        prev_scores = [model.data(model.index(i, 3)) for i in range(n)]

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual("AUC", pp.sfs.scorer.method)

        # change score to CA
        simulate.combobox_activate_item(self.widget.controls.scoring_method, "CA")
        self.wait_until_finished()
        self.assertEqual("CA", self.widget.stepwise_fs.scorer.method)
        new_scores = [model.data(model.index(i, 3)) for i in range(n)]
        self.assertNotEqual(prev_scores, new_scores)

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual("CA", pp.sfs.scorer.method)

    def test_validation_changed(self):
        model = self.widget.features_model
        n = len(self.iris.domain.attributes)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        # default validation is cross validation
        ex = ("Cross validation", {"k": 5})
        self.assertTupleEqual(ex, self.widget.stepwise_fs.scorer.validation)
        prev_scores = [model.data(model.index(i, 3)) for i in range(n)]
        self.assertEqual(0, self.widget.val_stacked_widget.currentIndex())
        control = self.widget.controls.cv_num_folds
        self.assertEqual(1, control.singleStep())
        self.assertEqual(2, control.minimum())
        self.assertEqual(10, control.maximum())

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(ex, pp.sfs.scorer.validation)

        # change validation to Random split
        simulate.combobox_activate_item(
            self.widget.controls.validation_method, "Random split"
        )
        self.wait_until_finished()
        ex = ("Random split", {"test_size": 0.1})
        self.assertTupleEqual(ex, self.widget.stepwise_fs.scorer.validation)

        new_scores = [model.data(model.index(i, 3)) for i in range(n)]
        self.assertNotEqual(prev_scores, new_scores)
        self.assertEqual(1, self.widget.val_stacked_widget.currentIndex())
        control = self.widget.controls.rs_test_size
        self.assertEqual(5, control.singleStep())
        self.assertEqual(1, control.minimum())
        self.assertEqual(99, control.maximum())

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(ex, pp.sfs.scorer.validation)

    def test_validation_parameter_changed(self):
        model = self.widget.features_model
        scorer = self.widget.stepwise_fs.scorer
        n = len(self.iris.domain.attributes)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        # default validation is cross validation with 5 folds
        ex = ("Cross validation", {"k": 5})
        self.assertTupleEqual(ex, scorer.validation)
        prev_scores = [model.data(model.index(i, 3)) for i in range(n)]
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(ex, pp.sfs.scorer.validation)

        # change number folds
        self.widget.controls.cv_num_folds.setValue(3)
        self.wait_until_finished()
        ex = ("Cross validation", {"k": 3})
        self.assertTupleEqual(ex, scorer.validation)
        new_scores = [model.data(model.index(i, 3)) for i in range(n)]
        self.assertNotEqual(prev_scores, new_scores)
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(ex, pp.sfs.scorer.validation)

        # change validation to Random split
        simulate.combobox_activate_item(
            self.widget.controls.validation_method, "Random split"
        )
        self.wait_until_finished()
        ex = ("Random split", {"test_size": 0.1})
        self.assertTupleEqual(ex, scorer.validation)
        prev_scores = [model.data(model.index(i, 3)) for i in range(n)]
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(ex, pp.sfs.scorer.validation)

        # change test split
        self.widget.controls.rs_test_size.setValue(20)
        self.wait_until_finished()
        ex = ("Random split", {"test_size": 0.2})
        self.assertTupleEqual(ex, scorer.validation)
        new_scores = [model.data(model.index(i, 3)) for i in range(n)]
        self.assertNotEqual(prev_scores, new_scores)
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(ex, pp.sfs.scorer.validation)

    def test_change_direction(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.widget.step_btn.click()
        self.wait_until_finished()

        # default direction Forward
        self.assertEqual("Forward", self.widget.stepwise_fs.direction)
        exp = [0.001, -0.00, "", 0.004]
        self.assert_list_almost_equal(exp, self.scores())
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual("Forward", pp.sfs.direction)

        # change direction to backward
        simulate.combobox_activate_item(self.widget.controls.direction, "Backward")
        self.wait_until_finished()
        self.assertEqual("Backward", self.widget.stepwise_fs.direction)
        self.assert_list_almost_equal(["", "", -0.493, ""], self.scores())
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual("Backward", pp.sfs.direction)

    def test_step_forward(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        self.assert_list_almost_equal([0.37, 0.26, 0.49, 0.49], self.scores())
        self.assert_list_almost_equal([0.5, 0.33, 0.17, 0.11, 0.33], self.stats())
        self.assertEqual(0, self.widget.num_entered_attr)
        self.assert_bold([False, False, True, False])

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.00, -0.00, "", 0.00], self.scores())
        self.assert_list_almost_equal([0.99, 0.95, 0.95, 0.95, 0.95], self.stats())
        self.assertEqual(1, self.widget.num_entered_attr)
        self.assert_bold([False, False, False, True])

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.00, 0.00, "", ""], self.scores())
        self.assert_list_almost_equal([1.00, 0.96, 0.96, 0.96, 0.96], self.stats())
        self.assertEqual(2, self.widget.num_entered_attr)
        self.assert_bold([True, False, False, False])

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal(["", "", "", ""], self.scores())
        self.assert_list_almost_equal([1.00, 0.96, 0.96, 0.96, 0.96], self.stats())
        self.assertEqual(4, self.widget.num_entered_attr)
        self.assert_bold([False, False, False, False])

        # try extra click when already all features included - should not fail
        self.widget.step_btn.click()
        self.assertFalse(self.widget.Error.unexpected_error.is_shown())
        self.assertEqual(4, self.widget.num_entered_attr)
        self.assert_bold([False, False, False, False])

    def test_step_backward(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        # include all features
        for _ in range(4):
            self.widget.step_btn.click()
            self.wait_until_finished()

        simulate.combobox_activate_item(self.widget.controls.direction, "Backward")
        self.wait_until_finished()
        self.assert_list_almost_equal([-0.00, 0.00, -0.01, -0.00], self.scores())
        self.assert_list_almost_equal([1.00, 0.96, 0.96, 0.96, 0.96], self.stats())
        self.assertEqual(4, self.widget.num_entered_attr)
        self.assert_bold([False, True, False, False])

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([-0.00, "", -0.01, -0.00], self.scores())
        self.assert_list_almost_equal([1.00, 0.96, 0.96, 0.96, 0.96], self.stats())
        self.assertEqual(3, self.widget.num_entered_attr)
        self.assert_bold([True, False, False, False])

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal(["", "", -0.01, -0.00], self.scores())
        self.assert_list_almost_equal([1.00, 0.96, 0.96, 0.96, 0.96], self.stats())
        self.assertEqual(2, self.widget.num_entered_attr)
        self.assert_bold([False, False, False, True])

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal(["", "", -0.50, ""], self.scores())
        self.assert_list_almost_equal([0.99, 0.95, 0.95, 0.95, 0.95], self.stats())
        self.assertEqual(1, self.widget.num_entered_attr)
        self.assert_bold([False, False, True, False])

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal(["", "", "", ""], self.scores())
        self.assert_list_almost_equal([0.5, 0.33, 0.17, 0.11, 0.33], self.stats())
        self.assertEqual(0, self.widget.num_entered_attr)
        self.assert_bold([False, False, False, False])

        # try extra click when already all features included - should not fail
        self.widget.step_btn.click()
        self.assertFalse(self.widget.Error.unexpected_error.is_shown())
        self.assertEqual(0, self.widget.num_entered_attr)

    def test_undo(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertEqual(0, self.widget.num_entered_attr)

        # prepare sequence to undo
        self.widget.step_btn.click()  # make step forward
        self.wait_until_finished()
        simulate.combobox_activate_item(self.widget.controls.direction, "Backward")
        self.wait_until_finished()
        self.widget.step_btn.click()  # make step back
        self.wait_until_finished()
        self.set_locked(1, True)  # lock feature
        self.set_selected(0, True)  # select feature
        self.set_locked(1, False)  # unlock feature
        self.set_selected(0, False)  # unselect feature

        # just set back to forward that all scores are shown
        simulate.combobox_activate_item(self.widget.controls.direction, "Forward")
        self.wait_until_finished()

        initial_diffs = [0.37, 0.26, 0.49, 0.49]
        initial_stats = [0.5, 0.33, 0.17, 0.11, 0.33]

        # all features unselected, unlocked
        self.assert_list_almost_equal(initial_diffs, self.scores())
        self.assert_list_almost_equal(initial_stats, self.stats())
        self.assert_locked([False] * 4)
        self.assert_selected([False] * 4)
        self.assertEqual(0, self.widget.num_entered_attr)

        # undo unselect feature - feature 2 should be selected
        self.widget.undo_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal(["", 0.05, 0.13, 0.12], self.scores())
        self.assert_list_almost_equal([0.87, 0.74, 0.74, 0.74, 0.74], self.stats())
        self.assert_locked([False] * 4)
        self.assert_selected([True, False, False, False])
        self.assertEqual(1, self.widget.num_entered_attr)

        # undo unlock feature - feature 2 should be selected, feature 3 should be locked
        self.widget.undo_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal(["", 0.05, 0.13, 0.12], self.scores())
        self.assert_list_almost_equal([0.87, 0.74, 0.74, 0.74, 0.74], self.stats())
        self.assert_locked([False, True, False, False])
        self.assert_selected([True, False, False, False])
        self.assertEqual(1, self.widget.num_entered_attr)

        # undo select feature - feature 3 should be locked
        self.widget.undo_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.37, "", 0.49, 0.49], self.scores())
        self.assert_list_almost_equal(initial_stats, self.stats())
        self.assert_locked([False, True, False, False])
        self.assert_selected([False] * 4)
        self.assertEqual(0, self.widget.num_entered_attr)

        # undo lock feature
        self.widget.undo_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.37, "", 0.49, 0.49], self.scores())
        self.assert_list_almost_equal(initial_stats, self.stats())
        self.assert_locked([False] * 4)
        self.assert_selected([False] * 4)
        self.assertEqual(0, self.widget.num_entered_attr)

        # undo step backward - feature 0 should be selected
        self.widget.undo_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.00, 0.00, "", 0.004], self.scores())
        self.assert_list_almost_equal([0.99, 0.95, 0.95, 0.95, 0.95], self.stats())
        self.assert_locked([False] * 4)
        self.assert_selected([False, False, True, False])
        self.assertEqual(1, self.widget.num_entered_attr)

        # undo step forward
        self.widget.undo_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal(initial_diffs, self.scores())
        self.assert_list_almost_equal(initial_stats, self.stats())
        self.assert_locked([False] * 4)
        self.assert_selected([False] * 4)
        self.assertEqual(0, self.widget.num_entered_attr)

    def test_start(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.wait_until_finished()

        # default stopping rule is n-features - 5
        self.widget.start_btn.click()
        self.assertEqual("Stop", self.widget.start_btn.text())
        self.wait_until_finished()

        self.assertEqual("Start", self.widget.start_btn.text())
        exp = [0.00, 0.00, 0.00, 0.01, "", "", 0.00, "", 0.00, 0.00, "", 0.00, ""]
        self.assert_list_almost_equal(exp, self.scores())
        t, f = True, False
        exp = [f, f, f, f, t, t, f, t, f, f, t, f, t]
        self.assert_selected(exp)

    def test_stepping_rule_change(self):
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.wait_until_finished()
        stacked = self.widget.stopping_stacked_widget

        simulate.combobox_activate_item(
            self.widget.controls.stopping_rule, "Score delta"
        )
        # spin label and properties should change
        self.assertEqual(1, self.widget.stopping_stacked_widget.currentIndex())
        control = stacked.currentWidget().findChild(QDoubleSpinBox)
        self.assertEqual(0, control.minimum())
        self.assertEqual(1000, control.maximum())
        self.assertEqual(4, control.decimals())

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(("Score delta", {"threshold": 0.1}), pp.stopping_rule)

        control.setValue(0.015)
        self.widget.start_btn.click()
        self.wait_until_finished()

        # we expect 3 steps with this stopping rule
        exp = [False] * 5 + [True, False, False, False, False, True, False, True]
        self.assert_selected(exp)
        exp = ["LSTAT", "PTRATIO", "RM"]
        self.assertListEqual(exp, self.sorted_vars(self.widget.entered_features))

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(("Score delta", {"threshold": 0.015}), pp.stopping_rule)

        # change to  something with no params
        simulate.combobox_activate_item(
            self.widget.controls.stopping_rule, "Minimum BIC"
        )
        self.assertEqual(3, self.widget.stopping_stacked_widget.currentIndex())

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(("Minimum BIC", {}), pp.stopping_rule)

        # change back to N-features
        simulate.combobox_activate_item(
            self.widget.controls.stopping_rule, "N-features"
        )
        self.assertEqual(0, self.widget.stopping_stacked_widget.currentIndex())
        control = stacked.currentWidget().findChild(QSpinBox)
        self.assertEqual(1, control.minimum())
        n = len(self.housing.domain.attributes)
        self.assertEqual(n, control.maximum())

        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(("N-features", {"n_features": 5}), pp.stopping_rule)

    def test_stop(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        # scores are computing buttons should be disabled
        self.assertFalse(self.widget.start_btn.isEnabled())
        self.assertFalse(self.widget.undo_btn.isEnabled())
        self.assertFalse(self.widget.step_btn.isEnabled())

        self.wait_until_finished()
        self.widget.start_btn.click()
        # when start is clicked the start button remains enabled but changes text
        self.assertTrue(self.widget.start_btn.isEnabled())
        self.assertEqual("Stop", self.widget.start_btn.text())
        self.assertFalse(self.widget.undo_btn.isEnabled())
        self.assertFalse(self.widget.step_btn.isEnabled())
        time.sleep(0.05)
        self.widget.start_btn.click()  # stop
        self.wait_until_finished()

        self.assertTrue(self.widget.start_btn.isEnabled())
        self.assertEqual("Start", self.widget.start_btn.text())
        self.assertTrue(self.widget.undo_btn.isEnabled())
        self.assertTrue(self.widget.step_btn.isEnabled())
        self.assert_selected([False, False, True, False])
        self.assert_list_almost_equal([0.001, -0.00, "", 0.004], self.scores())

    def test_variable_entered(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(0, self.widget.num_entered_attr)

        self.set_selected(0, True)
        self.wait_until_finished()
        # if scores changes variable enter works
        self.assert_list_almost_equal(["", 0.06, 0.13, 0.12], self.scores())
        exp = ["sepal length"]
        self.assertListEqual(exp, self.sorted_vars(self.widget.entered_features))
        self.assertEqual(1, self.widget.num_entered_attr)

        self.set_selected(1, True)
        self.wait_until_finished()
        self.assert_list_almost_equal(["", "", 0.07, 0.07], self.scores())
        exp = ["sepal length", "sepal width"]
        self.assertListEqual(exp, self.sorted_vars(self.widget.entered_features))
        self.assertEqual(2, self.widget.num_entered_attr)

    def test_variable_excluded(self):
        w = self.widget
        self.send_signal(self.widget.Inputs.data, self.iris)

        for i in (0, 1, 3):
            self.set_selected(i, True)
        self.wait_until_finished()
        self.assert_list_almost_equal(["", "", 0.01, ""], self.scores())
        exp = ["petal width", "sepal length", "sepal width"]
        self.assertListEqual(exp, self.sorted_vars(w.entered_features))

        self.set_selected(3, False)
        self.wait_until_finished()
        # if scores changes variable enter works
        self.assert_list_almost_equal(["", "", 0.07, 0.07], self.scores())
        exp = ["sepal length", "sepal width"]
        self.assertListEqual(exp, self.sorted_vars(w.entered_features))

        self.set_selected(0, False)
        self.wait_until_finished()
        # if scores changes variable enter works
        self.assert_list_almost_equal([0.16, "", 0.23, 0.23], self.scores())
        self.assertListEqual(["sepal width"], self.sorted_vars(w.entered_features))

        self.set_selected(1, False)
        self.wait_until_finished()
        self.assert_list_almost_equal([0.37, 0.26, 0.49, 0.49], self.scores())
        self.assertListEqual([], self.sorted_vars(self.widget.entered_features))

    def test_variable_locked(self):
        w = self.widget
        self.send_signal(self.widget.Inputs.data, self.iris)

        self.set_locked(0, True)
        self.wait_until_finished()
        self.assertListEqual(["sepal length"], self.sorted_vars(w.locked_features))

        self.set_locked(1, True)
        self.wait_until_finished()
        exp = ["sepal length", "sepal width"]
        self.assertListEqual(exp, self.sorted_vars(w.locked_features))

    def test_variable_unlocked(self):
        w = self.widget
        self.send_signal(self.widget.Inputs.data, self.iris)

        for i in (0, 1):
            self.set_locked(i, True)
        self.wait_until_finished()

        exp = ["sepal length", "sepal width"]
        self.assertListEqual(exp, self.sorted_vars(w.locked_features))

        self.set_locked(0, False)
        self.wait_until_finished()
        self.assertListEqual(["sepal width"], self.sorted_vars(w.locked_features))

        self.set_locked(1, False)
        self.wait_until_finished()
        self.assertListEqual([], self.sorted_vars(w.locked_features))

    def test_sort_table(self):
        view, model = self.widget.features_table, self.widget.features_table.model()
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.set_locked(2, True)
        self.set_locked(0, True)
        self.wait_until_finished()

        # default sorting by domain order
        attributes = [a.name for a in self.iris.domain.attributes]
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        self.assertListEqual(attributes, data)
        self.assert_list_almost_equal(["", 0.26, "", 0.49], self.scores())
        self.assertTupleEqual((-1, 0), self.widget.sort_column_order)

        # change to descending
        view.horizontalHeader().setSortIndicator(2, Qt.DescendingOrder)
        view.horizontalHeader().sectionClicked.emit(2)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        self.assertListEqual(sorted(attributes, reverse=True), data)
        self.assert_list_almost_equal([0.26, "", 0.49, ""], self.scores())
        self.assertTupleEqual((2, 1), self.widget.sort_column_order)

        # simulate click to scores column
        view.horizontalHeader().setSortIndicator(3, Qt.DescendingOrder)
        view.horizontalHeader().sectionClicked.emit(3)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["petal width", "sepal width", "sepal length", "petal length"]
        self.assertListEqual(attributes, data)
        self.assert_list_almost_equal([0.49, 0.26, "", ""], self.scores())
        self.assertTupleEqual((3, 1), self.widget.sort_column_order)

        # and other way around
        view.horizontalHeader().setSortIndicator(3, Qt.AscendingOrder)
        view.horizontalHeader().sectionClicked.emit(3)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["sepal length", "petal length", "sepal width", "petal width"]
        self.assertListEqual(attributes, data)
        self.assert_list_almost_equal(["", "", 0.26, 0.49], self.scores())
        self.assertTupleEqual((3, 0), self.widget.sort_column_order)

        # simulate click on locked column
        view.horizontalHeader().setSortIndicator(0, Qt.DescendingOrder)
        view.horizontalHeader().sectionClicked.emit(0)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["sepal length", "petal length", "sepal width", "petal width"]
        self.assertListEqual(attributes, data)
        self.assertTupleEqual((0, 1), self.widget.sort_column_order)

        # and other way around
        view.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        view.horizontalHeader().sectionClicked.emit(0)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["sepal width", "petal width", "sepal length", "petal length"]
        self.assertListEqual(attributes[::1], data)
        self.assertTupleEqual((0, 0), self.widget.sort_column_order)

        # simulate click on locked column
        self.set_selected(1, True)
        self.set_selected(3, True)
        view.horizontalHeader().setSortIndicator(1, Qt.DescendingOrder)
        view.horizontalHeader().sectionClicked.emit(1)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["sepal width", "petal width", "sepal length", "petal length"]
        self.assertListEqual(attributes, data)
        self.assertTupleEqual((1, 1), self.widget.sort_column_order)

        # and other way around
        view.horizontalHeader().setSortIndicator(1, Qt.AscendingOrder)
        view.horizontalHeader().sectionClicked.emit(1)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["sepal length", "petal length", "sepal width", "petal width"]
        self.assertListEqual(attributes[::1], data)
        self.assertTupleEqual((1, 0), self.widget.sort_column_order)

    def test_sort_table_backward(self):
        """When backward stepping is enabled entered values should be on top"""
        view, model = self.widget.features_table, self.widget.features_table.model()
        self.send_signal(self.widget.Inputs.data, self.iris)
        simulate.combobox_activate_item(self.widget.controls.direction, "Backward")
        self.wait_until_finished()

        # default sorting by domain order
        attributes = [a.name for a in self.iris.domain.attributes]
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        self.assertListEqual(attributes, data)
        self.assert_list_almost_equal(["", "", "", ""], self.scores())
        self.assertTupleEqual((-1, 0), self.widget.sort_column_order)

        # simulate click to scores column
        view.horizontalHeader().setSortIndicator(3, Qt.DescendingOrder)
        view.horizontalHeader().sectionClicked.emit(3)
        self.set_selected(1, True)
        self.set_selected(3, True)
        view.horizontalHeader().sectionClicked.emit(1)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["sepal width", "petal width", "sepal length", "petal length"]
        # by coincidence alphabetical order is the same as order of scores
        self.assertListEqual(attributes, data)
        self.assertTupleEqual((1, 1), self.widget.sort_column_order)

        # and other way around
        view.horizontalHeader().setSortIndicator(3, Qt.AscendingOrder)
        view.horizontalHeader().sectionClicked.emit(1)
        data = [model.data(model.index(i, 2)) for i in range(model.rowCount())]
        attributes = ["petal width", "sepal width", "sepal length", "petal length"]
        self.assertListEqual(attributes, data)
        self.assertTupleEqual((1, 0), self.widget.sort_column_order)

    def test_output(self):
        outputs = self.widget.Outputs
        d = self.iris.domain

        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        pp = self.get_output(outputs.preprocessor)
        self.assertEqual(pp.sfs.direction, "Forward")
        self.assertEqual(pp.sfs.scorer.method, "R2")
        self.assertEqual(pp.sfs.scorer.validation, ("Cross validation", {"k": 5}))
        self.assertEqual(pp.stopping_rule, ("N-features", {"n_features": 5}))
        self.assertIsNone(pp.sfs.scorer.learner)
        self.assertIsNone(self.get_output(outputs.model))

        self.send_signal(self.widget.Inputs.data, self.iris)
        table = self.get_output(outputs.data)
        self.assertTupleEqual((), table.domain.attributes)
        self.assertTupleEqual((d["iris"],), table.domain.class_vars)
        self.assertEqual(len(self.iris), len(table))
        self.assertIsInstance(self.get_output(outputs.model), ConstantModel)

        pp = self.get_output(outputs.preprocessor)
        self.assertEqual(pp.sfs.direction, "Forward")
        self.assertEqual(pp.sfs.scorer.method, "AUC")
        self.assertEqual(pp.sfs.scorer.validation, ("Cross validation", {"k": 5}))
        self.assertEqual(pp.stopping_rule, ("N-features", {"n_features": 4}))
        self.assertIsNone(pp.sfs.scorer.learner)

        self.widget.step_btn.click()
        self.wait_until_finished()

        table = self.get_output(outputs.data)
        self.assertTupleEqual((d["petal length"],), table.domain.attributes)
        self.assertTupleEqual((d["iris"],), table.domain.class_vars)
        self.assertEqual(len(self.iris), len(table))
        self.assertIsInstance(
            self.get_output(outputs.model), LogisticRegressionClassifier
        )

        pp = self.get_output(outputs.preprocessor)
        self.assertEqual(pp.sfs.direction, "Forward")
        self.assertEqual(pp.sfs.scorer.method, "AUC")
        self.assertEqual(pp.sfs.scorer.validation, ("Cross validation", {"k": 5}))
        self.assertEqual(pp.stopping_rule, ("N-features", {"n_features": 4}))
        self.assertIsNone(pp.sfs.scorer.learner)

        # change data to housing
        d = self.housing.domain
        self.send_signal(self.widget.Inputs.data, self.housing)
        table = self.get_output(outputs.data)
        self.assertTupleEqual((), table.domain.attributes)
        self.assertTupleEqual((d["MEDV"],), table.domain.class_vars)
        self.assertEqual(len(self.housing), len(table))
        self.assertIsInstance(self.get_output(outputs.model), MeanModel)

        pp = self.get_output(outputs.preprocessor)
        self.assertEqual(pp.sfs.direction, "Forward")
        self.assertEqual(pp.sfs.scorer.method, "R2")
        self.assertEqual(pp.sfs.scorer.validation, ("Cross validation", {"k": 5}))
        self.assertEqual(pp.stopping_rule, ("N-features", {"n_features": 4}))
        self.assertIsNone(pp.sfs.scorer.learner)

        self.widget.step_btn.click()
        self.wait_until_finished()

        table = self.get_output(outputs.data)
        self.assertTupleEqual((d["LSTAT"],), table.domain.attributes)
        self.assertTupleEqual((d["MEDV"],), table.domain.class_vars)
        self.assertEqual(len(self.housing), len(table))
        self.assertIsInstance(self.get_output(outputs.model), LinearModel)

        pp = self.get_output(outputs.preprocessor)
        self.assertEqual(pp.sfs.direction, "Forward")
        self.assertEqual(pp.sfs.scorer.method, "R2")
        self.assertEqual(pp.sfs.scorer.validation, ("Cross validation", {"k": 5}))
        self.assertEqual(pp.stopping_rule, ("N-features", {"n_features": 4}))
        self.assertIsNone(pp.sfs.scorer.learner)

        # remove data from the input
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(outputs.data))
        pp = self.get_output(self.widget.Outputs.preprocessor)
        self.assertEqual(pp.sfs.direction, "Forward")
        self.assertEqual(pp.sfs.scorer.method, "R2")
        self.assertEqual(pp.sfs.scorer.validation, ("Cross validation", {"k": 5}))
        self.assertEqual(pp.stopping_rule, ("N-features", {"n_features": 4}))
        self.assertIsNone(pp.sfs.scorer.learner)
        self.assertIsNone(self.get_output(outputs.model))

    def test_locked_not_enter(self):
        """Test that locked variable doesn't enter when stepping"""
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        self.set_locked(0, True)
        for _ in range(4):
            self.widget.step_btn.click()
            self.wait_until_finished()
        self.assert_selected([False, True, True, True])

    def test_unexpected_exception(self):
        path = (
            "orangecontrib.experiment_analytics.widgets.owstepwisefeatureselection."
            "StepwiseFeatureSelection.step"
        )

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        with patch(path, side_effect=ValueError("Test")):
            self.widget.step_btn.click()
            self.wait_until_finished()
        self.assertTrue(self.widget.Error.unexpected_error.is_shown())
        self.assertEqual("Test", str(self.widget.Error.unexpected_error))

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.unexpected_error.is_shown())

        with patch(path, side_effect=ValueError("Test")):
            self.widget.step_btn.click()
            self.wait_until_finished()
        self.assertTrue(self.widget.Error.unexpected_error.is_shown())

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.unexpected_error.is_shown())

    def test_incompatible_learner(self):
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.unexpected_error.is_shown())

        self.send_signal(self.widget.Inputs.learner, LinearRegressionLearner())
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.unexpected_error.is_shown())
        self.assertEqual(
            "Linear regression doesn't support categorical class variable. "
            "Please use different learner instead",
            str(self.widget.Error.unexpected_error),
        )
        self.assertFalse(self.widget.start_btn.isEnabled())
        self.assertFalse(self.widget.undo_btn.isEnabled())
        self.assertFalse(self.widget.step_btn.isEnabled())

        self.send_signal(self.widget.Inputs.learner, LogisticRegressionLearner())
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.unexpected_error.is_shown())
        self.assertTrue(self.widget.start_btn.isEnabled())
        self.assertTrue(self.widget.undo_btn.isEnabled())
        self.assertTrue(self.widget.step_btn.isEnabled())

        self.send_signal(self.widget.Inputs.data, self.housing)
        self.wait_until_finished()
        self.assertTrue(self.widget.Error.unexpected_error.is_shown())
        self.assertEqual(
            "Logistic regression doesn't support numeric class variable. "
            "Please use different learner instead",
            str(self.widget.Error.unexpected_error),
        )
        self.assertFalse(self.widget.start_btn.isEnabled())
        self.assertFalse(self.widget.undo_btn.isEnabled())
        self.assertFalse(self.widget.step_btn.isEnabled())

        self.send_signal(self.widget.Inputs.learner, LinearRegressionLearner())
        self.wait_until_finished()
        self.assertFalse(self.widget.Error.unexpected_error.is_shown())
        self.assertTrue(self.widget.start_btn.isEnabled())
        self.assertTrue(self.widget.undo_btn.isEnabled())
        self.assertTrue(self.widget.step_btn.isEnabled())

    def test_available_scores(self):
        def get_scores():
            cb = self.widget.scoring_method_cb
            return [cb.itemText(i) for i in range(cb.count())]

        # by default all scores are shown in dropdown
        self.assertListEqual(
            ["R2", "RMSE", "MAE", "AUC", "CA", "F1", "Prec", "Recall"], get_scores()
        )

        # classification learner
        self.send_signal(self.widget.Inputs.learner, LogisticRegressionLearner())
        self.assertListEqual(["AUC", "CA", "F1", "Prec", "Recall"], get_scores())

        # regression learner
        self.send_signal(self.widget.Inputs.learner, LinearRegressionLearner())
        self.assertListEqual(["R2", "RMSE", "MAE"], get_scores())

        # learner that support both regression and classification
        self.send_signal(self.widget.Inputs.learner, RandomForestLearner())
        self.assertListEqual(
            ["R2", "RMSE", "MAE", "AUC", "CA", "F1", "Prec", "Recall"], get_scores()
        )

        # data for classification
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertListEqual(["AUC", "CA", "F1", "Prec", "Recall"], get_scores())

        # data for regression
        self.send_signal(self.widget.Inputs.data, self.housing)
        self.assertListEqual(["R2", "RMSE", "MAE"], get_scores())

        # no data no learner all scoring methods should be available
        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.learner, None)
        self.assertListEqual(
            ["R2", "RMSE", "MAE", "AUC", "CA", "F1", "Prec", "Recall"], get_scores()
        )

    def test_titanic(self):
        self.send_signal(self.widget.Inputs.data, Table("titanic"))
        self.wait_until_finished()
        self.assert_list_almost_equal([0.13, 0.02, 0.19], self.scores())

    def test_nan_in_feature(self):
        with self.iris.unlocked(self.iris.X):
            self.iris[0, "petal length"] = np.nan

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assert_list_almost_equal([0.37, 0.26, 0.49, 0.49], self.scores())

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.00, 0.00, 0.00, ""], self.scores())

    def test_nan_class(self):
        with self.iris.unlocked(self.iris.Y):
            self.iris[0, "iris"] = np.nan

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertTrue(self.widget.Warning.nan_class.is_shown())
        self.assertEqual(149, len(self.widget.data))
        self.assert_list_almost_equal([0.37, 0.26, 0.49, 0.49], self.scores())

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.00, 0.00, "", 0.00], self.scores())

        # send data without nan in the class to check if warning disappear
        self.send_signal(self.widget.Inputs.data, self.iris[1:])
        self.wait_until_finished()
        self.assertFalse(self.widget.Warning.nan_class.is_shown())

    def test_nan_whole_column(self):
        nan_iris = self.iris.copy()
        with nan_iris.unlocked(nan_iris.X):
            nan_iris[:, "petal length"] = np.nan

        self.send_signal(self.widget.Inputs.data, nan_iris)
        self.wait_until_finished()
        exp = ["petal width", "sepal length", "sepal width"]
        self.assertListEqual(exp, self.sorted_vars(self.widget.data.domain.attributes))
        self.assert_list_almost_equal([0.37, 0.26, 0.49], self.scores())
        self.assertTrue(self.widget.Warning.nan_col.is_shown())

        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assert_list_almost_equal([0.00, 0.00, ""], self.scores())

        # send data without nan in the class to check if warning disappear
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertFalse(self.widget.Warning.nan_col.is_shown())

    def test_no_features(self):
        data = self.iris.transform(Domain([], self.iris.domain.class_vars))
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()

        self.assertEqual(0, self.widget.features_model.rowCount())
        self.assertEqual(0, self.widget.num_all_attr)

    @staticmethod
    def click_header_section(view, section):
        header = view.horizontalHeader()
        p = header.rect().center()
        p.setX(header.sectionPosition(section) + 5)
        QTest.mouseClick(header.viewport(), Qt.LeftButton, pos=p)

    def test_lock_all(self):
        """Test lock all functionality in table's header"""
        d = self.iris.domain
        w = self.widget

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertListEqual([], self.widget.locked_features)
        self.assertSetEqual(set(), self.widget.stepwise_fs.locked)

        # when all unlocked click on the header will lock all
        self.click_header_section(self.widget.features_table, 0)
        all_attrs = self.sorted_vars(d.attributes)
        self.assertListEqual(all_attrs, self.sorted_vars(w.locked_features))
        self.assertSetEqual(set(d.attributes), self.widget.stepwise_fs.locked)

        # another click unlock all
        self.click_header_section(self.widget.features_table, 0)
        self.assertListEqual([], self.widget.locked_features)
        self.assertSetEqual(set(), self.widget.stepwise_fs.locked)

        # when at least one is locked click will lock all
        self.set_locked(0, True)
        self.assertListEqual(["sepal length"], self.sorted_vars(w.locked_features))
        self.assertSetEqual({d["sepal length"]}, self.widget.stepwise_fs.locked)
        self.click_header_section(self.widget.features_table, 0)
        self.assertListEqual(all_attrs, self.sorted_vars(w.locked_features))
        self.assertSetEqual(set(d.attributes), self.widget.stepwise_fs.locked)

        # test no data
        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()
        self.assertListEqual([], self.widget.locked_features)
        self.assertSetEqual(set(), self.widget.stepwise_fs.locked)
        self.click_header_section(self.widget.features_table, 0)

    def test_select_all(self):
        """Test select all functionality in table's header"""
        d = self.iris.domain
        w = self.widget
        all_attrs = self.sorted_vars(d.attributes)
        all_set = set(d.attributes)

        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.assertListEqual([], self.widget.entered_features)
        self.assertSetEqual(set(), self.widget.stepwise_fs.selected)
        self.assertEqual(0, self.widget.num_entered_attr)

        # when all unselected click on the header will select all
        self.click_header_section(self.widget.features_table, 1)
        self.wait_until_finished()
        self.assertListEqual(all_attrs, self.sorted_vars(w.entered_features))
        self.assertSetEqual(all_set, self.widget.stepwise_fs.selected)
        self.assertEqual(len(d.attributes), self.widget.num_entered_attr)

        # another click unselect all
        self.click_header_section(self.widget.features_table, 1)
        self.wait_until_finished()
        self.assertListEqual([], self.widget.entered_features)
        self.assertSetEqual(set(), self.widget.stepwise_fs.selected)
        self.assertEqual(0, self.widget.num_entered_attr)

        # when at least one is selected click will select all
        self.widget.step_btn.click()
        self.wait_until_finished()
        self.assertListEqual(["petal length"], self.sorted_vars(w.entered_features))
        self.assertSetEqual({d["petal length"]}, self.widget.stepwise_fs.selected)
        self.click_header_section(self.widget.features_table, 1)
        self.wait_until_finished()
        self.assertListEqual(all_attrs, self.sorted_vars(w.entered_features))
        self.assertSetEqual(all_set, self.widget.stepwise_fs.selected)
        self.assertEqual(len(d.attributes), self.widget.num_entered_attr)

        # test locked feature should not be unselected
        self.set_locked(0, True)
        self.click_header_section(self.widget.features_table, 1)
        self.wait_until_finished()
        self.assertListEqual(["sepal length"], self.sorted_vars(w.entered_features))
        self.assertSetEqual({d["sepal length"]}, self.widget.stepwise_fs.selected)
        self.assertEqual(1, self.widget.num_entered_attr)

        # test locked features should not be selected
        self.set_locked(0, False)
        self.click_header_section(self.widget.features_table, 1)
        self.wait_until_finished()
        self.click_header_section(self.widget.features_table, 1)
        self.wait_until_finished()
        self.assertListEqual([], self.widget.entered_features)
        self.assertSetEqual(set(), self.widget.stepwise_fs.selected)
        self.assertEqual(0, self.widget.num_entered_attr)

        self.set_locked(0, True)
        self.click_header_section(self.widget.features_table, 1)
        self.wait_until_finished()
        attributes = self.sorted_vars(d.attributes[1:])
        self.assertListEqual(attributes, self.sorted_vars(w.entered_features))
        self.assertSetEqual(all_set - {d["sepal length"]}, w.stepwise_fs.selected)
        self.assertEqual(len(d.attributes) - 1, self.widget.num_entered_attr)

        # test no data
        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()
        self.assertListEqual([], self.widget.locked_features)
        self.assertSetEqual(set(), self.widget.stepwise_fs.locked)
        self.click_header_section(self.widget.features_table, 0)
        self.assertEqual(0, self.widget.num_entered_attr)
        self.wait_until_finished()

    @patch(f"{WIDGET_PATH}.report_table")
    @patch(f"{WIDGET_PATH}.report_items")
    def test_report_no_data(self, report_item_mock, report_table_mock):
        self.widget.report_button.click()
        report_table_mock.assert_not_called()
        exp = {
            "Validation method": "Cross validation",
            "&nbsp;&nbsp;Number folds": 5,
            "Score": "R2",
            "Direction": "Forward",
            "Stopping rule": "N-features: 5",
        }
        report_item_mock.assert_called_with("Settings", exp)
        report_item_mock.reset_mock()

        simulate.combobox_activate_item(
            self.widget.controls.validation_method, "Random split"
        )
        simulate.combobox_activate_item(
            self.widget.controls.stopping_rule, "Score delta"
        )
        self.widget.report_button.click()
        report_table_mock.assert_not_called()
        exp = {
            "Validation method": "Random split",
            "&nbsp;&nbsp;Test set size": 10,
            "Score": "R2",
            "Direction": "Forward",
            "Stopping rule": "Score delta: 0.1",
        }
        report_item_mock.assert_called_with("Settings", exp)
        report_item_mock.reset_mock()

        simulate.combobox_activate_item(
            self.widget.controls.stopping_rule, "Minimum BIC"
        )
        self.widget.report_button.click()
        report_table_mock.assert_not_called()
        exp = {
            "Validation method": "Random split",
            "&nbsp;&nbsp;Test set size": 10,
            "Score": "R2",
            "Direction": "Forward",
            "Stopping rule": "Minimum BIC",
        }
        report_item_mock.assert_called_with("Settings", exp)
        report_item_mock.reset_mock()

        simulate.combobox_activate_item(
            self.widget.controls.stopping_rule, "Minimum AICc"
        )
        self.widget.report_button.click()
        report_table_mock.assert_not_called()
        exp = {
            "Validation method": "Random split",
            "&nbsp;&nbsp;Test set size": 10,
            "Score": "R2",
            "Direction": "Forward",
            "Stopping rule": "Minimum AICc",
        }
        report_item_mock.assert_called_with("Settings", exp)
        report_item_mock.reset_mock()

    @patch(f"{WIDGET_PATH}.report_table")
    @patch(f"{WIDGET_PATH}.report_items")
    def test_report_data(self, report_item_mock, report_table_mock):
        self.send_signal(self.widget.Inputs.data, self.iris)

        self.widget.report_button.click()
        exp = {
            "Validation method": "Cross validation",
            "&nbsp;&nbsp;Number folds": 5,
            "Score": "AUC",
            "Direction": "Forward",
            "Stopping rule": "N-features: 4",
        }
        report_item_mock.assert_called_with("Settings", exp)
        report_item_mock.reset_mock()
        self.assertEqual(report_table_mock.call_count, 2)
        exp = [
            ["Locked", "Entered", "Feature", "Score difference"],
            ["&cross;", "&cross;", "sepal length", ANY],
            ["&cross;", "&cross;", "sepal width", ANY],
            ["&cross;", "&cross;", "petal length", ANY],
            ["&cross;", "&cross;", "petal width", ANY],
        ]
        report_table_mock.assert_called_with("Current estimates", exp, header_rows=1)

        view = self.widget.features_table
        view.horizontalHeader().setSortIndicator(2, Qt.AscendingOrder)
        view.horizontalHeader().sectionClicked.emit(2)
        self.widget.report_button.click()
        exp = [
            ["Locked", "Entered", "Feature", "Score difference"],
            ["&cross;", "&cross;", "petal length", ANY],
            ["&cross;", "&cross;", "petal width", ANY],
            ["&cross;", "&cross;", "sepal length", ANY],
            ["&cross;", "&cross;", "sepal width", ANY],
        ]
        report_table_mock.assert_called_with("Current estimates", exp, header_rows=1)

    def test_transformation_added(self):
        # table doesn't have transformations - attribute no transformation is added
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.widget.step_btn.click()
        output = self.get_output(self.widget.Outputs.data)
        self.assertNotIn(TRANSFORMATIONS_ATTRIBUTE, output.attributes)

        # table has transformations - add domain and aggregation preprocessing
        ti = InfoTransform(self.iris.domain)
        self.iris.attributes[TRANSFORMATIONS_ATTRIBUTE] = (ti,)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.wait_until_finished()
        self.widget.step_btn.click()
        output = self.get_output(self.widget.Outputs.data)
        transformations = output.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(transformations))

        # info transform also in the beginning of workflow
        self.assertIsInstance(transformations[0], InfoTransform)
        self.assertEqual(self.iris.domain, transformations[0].domain)

        # test stepwise transformation
        self.assertIsInstance(transformations[1], StepwiseFeatureSelectionTransform)
        self.assertEqual(output.domain, transformations[1].domain)

    def test_input_similar_domain_data(self):
        data = self.iris
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.start_btn.click()
        self.wait_until_finished()

        attrs = data.domain.attributes
        domain = Domain(attrs[:-1], attrs[-1])
        data = data.transform(domain)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()

        self.assertNotIn(attrs[-1], self.widget.entered_features)
        self.assertTrue(self.widget.start_btn.isEnabled())

    def test_saved_workflow(self):
        data = self.iris
        self.send_signal(self.widget.Inputs.data, data)
        self.set_locked(3, True)
        self.set_selected(0, True)
        self.set_selected(1, True)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        widget = self.create_widget(OWStepwiseFeatureSelection,
                                    stored_settings=settings)
        self.send_signal(widget.Inputs.data, data, widget=widget)
        self.assertEqual(widget.entered_features, self.widget.entered_features)
        self.assertEqual(widget.locked_features, self.widget.locked_features)

        self.send_signal(widget.Inputs.data, self.housing, widget=widget)
        self.assertEqual(widget.entered_features, [])
        self.assertEqual(widget.locked_features, [])

    def test_reload_data(self):
        data = self.iris
        self.send_signal(self.widget.Inputs.data, data)
        self.set_locked(3, True)
        self.set_selected(0, True)
        self.set_selected(1, True)
        entered_features = self.widget.entered_features.copy()
        locked_features = self.widget.locked_features.copy()

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(entered_features, self.widget.entered_features)
        self.assertEqual(locked_features, self.widget.locked_features)

    def test_retain_metas(self):
        data = Table("zoo")
        data = data.add_column(ContinuousVariable("nan"),
                               np.full(len(data), np.nan))
        self.send_signal(self.widget.Inputs.data, data)
        self.set_selected(1, True)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual([m.name for m in output.domain.metas], ["name"])


class TestSlicerPreprocessor(unittest.TestCase):
    def setUp(self):
        self.data = Table("iris")

    def test_repr_features_selected(self):
        sfst = StepwiseFeatureSelectionTransform(self.data.domain)
        self.assertEqual(
            f"<h4>Stepwise Feature Selection</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Entered features: </th><td>sepal length, sepal width, "
            "petal length, petal width</td></tr></table></div>",
            str(sfst),
        )

        domain = Domain(self.data.domain.attributes[:2], self.data.domain.class_vars)
        sfst = StepwiseFeatureSelectionTransform(domain)
        self.assertEqual(
            f"<h4>Stepwise Feature Selection</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Entered features: </th>"
            "<td>sepal length, sepal width</td></tr></table></div>",
            str(sfst),
        )

    def test_repr_no_features_selected(self):
        domain = Domain([], self.data.domain.class_vars)
        sfst = StepwiseFeatureSelectionTransform(domain)
        self.assertEqual(
            f"<h4>Stepwise Feature Selection</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Entered features: </th><td>(none)</td></tr></table></div>",
            str(sfst),
        )


if __name__ == "__main__":
    unittest.main()
