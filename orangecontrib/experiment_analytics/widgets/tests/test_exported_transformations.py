import os.path
import pickle
import unittest
from typing import List, Tuple, Any

import numpy as np
from AnyQt.QtCore import QItemSelectionModel
from AnyQt.QtWidgets import QListView
from numpy.testing import assert_array_almost_equal, assert_array_equal
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
from Orange.preprocess.transformation import Normalizer
from Orange.widgets.data.oweditdomain import (
    AsCategorical,
    OWEditDomain,
    Rename,
    TransformRole,
)
from Orange.widgets.data.owpreprocess import OWPreprocess, Scale
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    InfoTransform,
)
from orangecontrib.experiment_analytics.widgets.owaggregate import OWAggregate
from orangecontrib.experiment_analytics.widgets.owinitializetransformation import (
    OWInitializeTransformation,
)
from orangecontrib.experiment_analytics.widgets.owsavetransformations import OWSaveTransformations
from orangecontrib.experiment_analytics.widgets.owslicer import OWSlicer
from orangecontrib.experiment_analytics.widgets.owstepwisefeatureselection import (
    OWStepwiseFeatureSelection,
)
from orangecontrib.experiment_analytics.widgets.tests.test_owsavetransformations import random_temp_file


class TestExportScenarios(WidgetTest):
    def setUp(self):
        self.export_w = self.create_widget(OWSaveTransformations)
        self.init_w = self.create_widget(OWInitializeTransformation)

        domain = Domain(
            [ContinuousVariable("a"), ContinuousVariable("b")],
            [ContinuousVariable("class")],
        )
        x = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [1, 3], [2, 5]])
        self.data = Table.from_numpy(domain, x, x[:, 0])

        domain = Domain(
            [ContinuousVariable("a"), ContinuousVariable("b"), ContinuousVariable("c")],
            [ContinuousVariable("class")],
        )
        x = [
            [2, 2, 1],
            [1, 1, 1],
            [1, 2, 1],
            [3, 5, 1],
            [2, 3, 1],
            [5, 2, 1],
            [4, 3, 1],
            [4, 2, 1],
        ]
        self.data2 = Table.from_numpy(domain, x, np.arange(0, len(x)).reshape(-1, 1))

    @staticmethod
    def apply_transformation(pickle_path: str, data: Table) -> Table:
        with open(pickle_path, "rb") as f:
            return pickle.load(f)(data)

    def apply_init(self, in_data: Table) -> Table:
        self.send_signal(self.init_w.Inputs.data, in_data, widget=self.init_w)
        out_data = self.get_output(self.init_w.Outputs.data, widget=self.init_w)
        self.assertNotIn(TRANSFORMATIONS_ATTRIBUTE, in_data.attributes)
        self.assertTupleEqual(
            (InfoTransform(self.data.domain),),
            out_data.attributes[TRANSFORMATIONS_ATTRIBUTE],
        )
        return out_data

    def apply_export(self, in_data: Table, path: str):
        self.send_signal(self.export_w.Inputs.data, in_data, widget=self.export_w)
        self.export_w.filename = path
        self.export_w.bt_save.click()
        self.assertTrue(os.path.exists(path))

    def apply_normalize(self, in_data: Table) -> Table:
        widget = self.create_widget(OWPreprocess)
        saved = {
            "preprocessors": [
                ("orange.preprocess.scale", {"method": Scale.NormalizeBySpan_ZeroBased})
            ]
        }
        widget.set_model(widget.load(saved))
        self.send_signal(widget.Inputs.data, in_data, widget=widget)
        return self.get_output(widget.Outputs.preprocessed_data, widget=widget)

    def apply_edit_domain(
        self, in_data: Table, transformations: Tuple[Tuple[int, Any], ...]
    ) -> Table:
        """Rename and change type variable transformation"""
        widget = self.create_widget(OWEditDomain)
        model = widget.domain_view.model()
        self.send_signal(widget.Inputs.data, in_data, widget=widget)
        for idx, trans in transformations:
            model.setData(model.index(idx, 0), [trans], TransformRole)
        widget.commit()
        return self.get_output(widget.Outputs.data, widget=widget)

    def apply_slicer(
        self,
        in_data: Table,
        variable: str,
        selection: List[Tuple[Tuple[float, float], str]],
    ) -> Table:
        widget = self.create_widget(OWSlicer)
        self.send_signal(widget.Inputs.data, in_data, widget)
        simulate.combobox_activate_item(widget.controls.x_var, variable)
        widget.selection = selection
        widget.commit.deferred()
        return self.get_output(widget.Outputs.selected_data, widget=widget)

    @staticmethod
    def __set_selection(view: QListView, indices: List[int]):
        view.clearSelection()
        sm = view.selectionModel()
        model = view.model()
        for ind in indices:
            sm.select(model.index(ind, 0), QItemSelectionModel.Select)

    def apply_aggregation(
        self, in_data: Table, row_idx: List[int], value_idx: List[int]
    ) -> Table:
        widget = self.create_widget(OWAggregate)
        self.send_signal(widget.Inputs.data, in_data, widget)
        self.__set_selection(widget.row_attrs_view, row_idx)
        self.__set_selection(widget.val_attrs_view, value_idx)
        widget.aggregation_cbs["Sum"].click()
        return self.get_output(widget.Outputs.data, widget=widget)

    def apply_stepwise_feature_selection(self, in_data: Table) -> Table:
        widget = self.create_widget(OWStepwiseFeatureSelection)
        self.send_signal(widget.Inputs.data, in_data, widget)
        self.wait_until_finished(widget=widget)
        widget.step_btn.click()
        return self.get_output(widget.Outputs.data, widget=widget)

    def test_simple_example(self):
        data = self.apply_init(self.data)

        with random_temp_file() as path:
            self.apply_export(data, path)
            # nothing should happen sine no real transformation on data
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)
        self.assertEqual(self.data.domain, result.domain)
        assert_array_equal(self.data.X, result.X)

        ex = (ContinuousVariable("a"), ContinuousVariable("b"))
        self.assertTupleEqual(ex, result2.domain.attributes)
        self.assertTupleEqual((ContinuousVariable("class"),), result2.domain.class_vars)
        expected = np.array(
            [[2, 2], [1, 1], [1, 2], [3, 5], [2, 3], [5, 2], [4, 3], [4, 2]]
        )
        assert_array_equal(expected, result2.X)

    def test_only_orange_transformations(self):
        data = self.apply_init(self.data)
        data = self.apply_normalize(data)

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        assert_array_equal(data.X, result.X)
        expected = np.array(
            [[0, 0], [0.33, 0.33], [0.66, 0.66], [1, 1], [0, 0.33], [0.33, 1]]
        )
        assert_array_almost_equal(expected, result.X, decimal=2)

        self.assertEqual(data.domain, result2.domain)
        o3 = 0.3
        expected = np.array(
            [[o3, 0], [0, -o3], [0, 0], [0.6, 1], [o3, 0.3], [1.3, 0], [1, o3], [1, 0]]
        )
        assert_array_almost_equal(expected, result2.X, decimal=1)

    def test_only_orange_transformations2(self):
        data = self.apply_init(self.data)
        data = self.apply_edit_domain(data, ((0, Rename("foo")), (1, AsCategorical())))
        data = self.apply_normalize(data)

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        attrs = result.domain.attributes
        self.assertIsInstance(attrs[0], ContinuousVariable)
        self.assertIsInstance(attrs[1], DiscreteVariable)
        self.assertEqual("foo", attrs[0].name)
        self.assertEqual("b", attrs[1].name)
        self.assertTupleEqual(("2.0", "3.0", "4.0", "5.0"), result.domain[1].values)

        assert_array_equal(data.X, result.X)
        expected = np.array([[0, 0], [0.33, 1], [0.66, 2], [1, 3], [0, 1], [0.33, 3]])
        assert_array_almost_equal(expected, result.X, decimal=2)

        # test with data 2
        self.assertEqual(data.domain, result2.domain)
        attrs = result2.domain.attributes
        self.assertIsInstance(attrs[0], ContinuousVariable)
        self.assertIsInstance(attrs[1], DiscreteVariable)
        self.assertEqual("foo", attrs[0].name)
        self.assertEqual("b", attrs[1].name)
        self.assertTupleEqual(("2.0", "3.0", "4.0", "5.0"), result.domain[1].values)

        nan = np.nan
        expected = np.array(
            [[0.3, 0], [0, nan], [0, 0], [0.6, 3], [0.3, 1], [1.3, 0], [1, 1], [1, 0]]
        )
        assert_array_almost_equal(expected, result2.X, decimal=1)

    def test_slicer_transformation(self):
        data = self.apply_init(self.data)
        data = self.apply_slicer(data, "a", [((0, 2.1), "S1"), ((2.5, 10), "S2")])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        self.assertTupleEqual(
            (ContinuousVariable("a"), ContinuousVariable("b")), result.domain.attributes
        )
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)
        assert_array_equal(data.X, result.X)
        expected = np.array([[1, 2], [2, 3], [1, 3], [2, 5], [3, 4], [4, 5]])
        assert_array_equal(expected, result.X)
        assert_array_equal(np.array([[0], [0], [0], [0], [1], [1]]), result.metas)

        ex = (ContinuousVariable("a"), ContinuousVariable("b"))
        self.assertTupleEqual(ex, result2.domain.attributes)
        expected = np.array(
            [[2, 2], [1, 1], [1, 2], [2, 3], [3, 5], [5, 2], [4, 3], [4, 2]]
        )
        assert_array_equal(expected, result2.X)
        assert_array_equal(
            np.array([[0], [0], [0], [0], [1], [1], [1], [1]]), result2.metas
        )

    def test_aggregation_transformation(self):
        data = self.apply_init(self.data)
        data = self.apply_aggregation(data, [1], [2])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        expected = (
            ContinuousVariable("a"),
            ContinuousVariable("b - Mean"),
            ContinuousVariable("b - Sum"),
        )
        self.assertTupleEqual(expected, result.domain.attributes)

        assert_array_equal(data.X, result.X)
        expected = np.array([[1, 2.5, 5], [2, 4, 8], [3, 4, 4], [4, 5, 5]])
        assert_array_equal(expected, result.X)
        assert_array_equal(np.empty((4, 0)), result.metas)

        self.assertEqual(data.domain, result2.domain)
        expected = np.array(
            [[1, 1.5, 3], [2, 2.5, 5], [3, 5, 5], [4, 2.5, 5], [5, 2, 2]]
        )
        assert_array_equal(expected, result2.X)
        assert_array_equal(np.empty((5, 0)), result2.metas)

    def test_all_transformations(self):
        data = self.apply_init(self.data)
        data = self.apply_aggregation(data, [1], [2])
        data = self.apply_slicer(data, "a", [((0, 2.1), "S1"), ((2.5, 10), "S2")])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)

        self.assertEqual(data.domain, result.domain)
        expected = (
            ContinuousVariable("a"),
            ContinuousVariable("b - Mean"),
            ContinuousVariable("b - Sum"),
        )
        self.assertTupleEqual(expected, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        assert_array_equal(data.X, result.X)
        expected = np.array([[1, 2.5, 5], [2, 4, 8], [3, 4, 4], [4, 5, 5]])
        assert_array_equal(expected, result.X)
        assert_array_equal(np.array([[0], [0], [1], [1]]), result.metas)

    def test_with_orange_transformations1(self):
        data = self.apply_init(self.data)
        data = self.apply_normalize(data)
        data = self.apply_aggregation(data, [1], [2])
        data = self.apply_slicer(data, "a", [((0, 0.5), "S1"), ((0.5, 10), "S2")])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        expected = (
            ContinuousVariable("a"),
            ContinuousVariable("b - Mean"),
            ContinuousVariable("b - Sum"),
        )
        self.assertTupleEqual(expected, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        assert_array_equal(data.X, result.X)
        expected = np.array(
            [[0, 0.17, 0.33], [0.33, 0.66, 1.33], [0.66, 0.66, 0.66], [1, 1, 1]],
        )
        assert_array_almost_equal(expected, result.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1]]), result.metas)

        self.assertEqual(data.domain, result2.domain)
        expected = (
            ContinuousVariable("a"),
            ContinuousVariable("b - Mean"),
            ContinuousVariable("b - Sum"),
        )
        self.assertTupleEqual(expected, result2.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result2.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result2.domain.metas[0].values)

        assert_array_equal(data.X, result.X)
        expected = np.array(
            [
                [0, -0.17, -0.33],
                [0.33, 0.17, 0.33],
                [0.67, 1, 1],
                [1, 0.17, 0.33],
                [1.33, 0, 0],
            ],
        )
        assert_array_almost_equal(expected, result2.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1], [1]]), result2.metas)

    def test_with_orange_transformations2(self):
        data = self.apply_init(self.data)
        data = self.apply_aggregation(data, [1], [2])
        data = self.apply_normalize(data)
        data = self.apply_slicer(data, "a", [((0, 0.5), "S1"), ((0.5, 10), "S2")])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        expected_domain = (
            ContinuousVariable(
                "a", compute_value=Normalizer(ContinuousVariable("a"), 1, 1 / 3)
            ),
            ContinuousVariable(
                "b - Mean",
                compute_value=Normalizer(ContinuousVariable("b - Mean"), 2.5, 0.4),
            ),
            ContinuousVariable(
                "b - Sum",
                compute_value=Normalizer(ContinuousVariable("b - Sum"), 4, 1 / 4),
            ),
        )
        self.assertTupleEqual(expected_domain, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        assert_array_equal(data.X, result.X)
        expected = np.array(
            [[0, 0, 1 / 4], [0.33, 0.6, 1], [0.66, 0.6, 0], [1, 1, 1 / 4]],
        )
        assert_array_almost_equal(expected, result.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1]]), result.metas)

        self.assertEqual(data.domain, result2.domain)
        self.assertTupleEqual(expected_domain, result2.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result2.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result2.domain.metas[0].values)

        expected = np.array(
            [
                [0, -0.4, -1 / 4],
                [0.33, 0, 1 / 4],
                [0.67, 1, 1 / 4],
                [1, 0, 1 / 4],
                [1.33, -0.2, -2 / 4],
            ],
        )
        assert_array_almost_equal(expected, result2.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1], [1]]), result2.metas)

    def test_with_orange_transformations3(self):
        data = self.apply_init(self.data)
        data = self.apply_aggregation(data, [1], [2])
        data = self.apply_slicer(data, "a", [((0, 2.1), "S1"), ((2.5, 10), "S2")])
        data = self.apply_normalize(data)

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        expected_domain = (
            ContinuousVariable(
                "a", compute_value=Normalizer(ContinuousVariable("a"), 1, 1 / 3)
            ),
            ContinuousVariable(
                "b - Mean",
                compute_value=Normalizer(ContinuousVariable("b - Mean"), 2.5, 0.4),
            ),
            ContinuousVariable(
                "b - Sum",
                compute_value=Normalizer(ContinuousVariable("b - Sum"), 4, 1 / 4),
            ),
        )
        self.assertTupleEqual(expected_domain, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        assert_array_equal(data.X, result.X)
        expected = np.array(
            [[0, 0, 1 / 4], [0.33, 0.6, 1], [0.66, 0.6, 0], [1, 1, 1 / 4]],
        )
        assert_array_almost_equal(expected, result.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1]]), result.metas)

        # data 2
        self.assertEqual(data.domain, result2.domain)
        self.assertTupleEqual(expected_domain, result2.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result2.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result2.domain.metas[0].values)

        expected = np.array(
            [
                [0, -0.4, -1 / 4],
                [0.33, 0, 1 / 4],
                [0.67, 1, 1 / 4],
                [1, 0, 1 / 4],
                [1.33, -0.2, -2 / 4],
            ],
        )
        assert_array_almost_equal(expected, result2.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1], [1]]), result2.metas)

    def test_with_orange_transformations_double(self):
        data = self.apply_init(self.data)
        data = self.apply_normalize(data)
        data = self.apply_aggregation(data, [1], [2])
        data = self.apply_slicer(data, "a", [((0, 0.5), "S1"), ((0.5, 10), "S2")])
        data = self.apply_edit_domain(data, ((0, Rename("foo")), (1, AsCategorical())))

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        attrs = result.domain.attributes
        self.assertIsInstance(attrs[0], ContinuousVariable)
        self.assertIsInstance(attrs[1], DiscreteVariable)
        self.assertIsInstance(attrs[2], ContinuousVariable)
        self.assertEqual("foo", attrs[0].name)
        self.assertEqual("b - Mean", attrs[1].name)
        self.assertEqual("b - Sum", attrs[2].name)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)
        self.assertTupleEqual(
            ("0.16666666666666666", "0.6666666666666666", "1.0"), attrs[1].values
        )

        assert_array_equal(data.X, result.X)
        expected = np.array(
            [[0, 0, 0.33], [0.33, 1, 1.33], [0.66, 1, 0.66], [1, 2, 1]],
        )
        assert_array_almost_equal(expected, result.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1]]), result.metas)

        # data 2
        self.assertEqual(data.domain, result2.domain)
        attrs = result2.domain.attributes
        self.assertIsInstance(attrs[0], ContinuousVariable)
        self.assertIsInstance(attrs[1], DiscreteVariable)
        self.assertIsInstance(attrs[2], ContinuousVariable)
        self.assertEqual("foo", attrs[0].name)
        self.assertEqual("b - Mean", attrs[1].name)
        self.assertEqual("b - Sum", attrs[2].name)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result2.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result2.domain.metas[0].values)

        expected = np.array(
            [
                [0, np.nan, -0.33],
                [0.33, 0, 0.33],
                [0.67, 2, 1],
                [1, 0, 0.33],
                [1.33, np.nan, 0],
            ],
        )
        assert_array_almost_equal(expected, result2.X, decimal=2)
        assert_array_equal(np.array([[0.0], [0.0], [1.0], [1.0], [1.0]]), result2.metas)

    def test_with_orange_transformations_double2(self):
        data = self.apply_init(self.data)
        data = self.apply_aggregation(data, [1], [2])
        data = self.apply_normalize(data)
        data = self.apply_edit_domain(data, ((0, Rename("foo")), (1, AsCategorical())))
        data = self.apply_slicer(data, "foo", [((0, 0.5), "S1"), ((0.5, 10), "S2")])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        attrs = result.domain.attributes
        self.assertIsInstance(attrs[0], ContinuousVariable)
        self.assertIsInstance(attrs[1], DiscreteVariable)
        self.assertIsInstance(attrs[2], ContinuousVariable)
        self.assertEqual("foo", attrs[0].name)
        self.assertEqual("b - Mean", attrs[1].name)
        self.assertEqual("b - Sum", attrs[2].name)
        self.assertTupleEqual(("0.0", "0.6000000000000001", "1.0"), attrs[1].values)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        assert_array_equal(data.X, result.X)
        expected = np.array(
            [[0, 0, 1 / 4], [0.33, 1, 1], [0.66, 1, 0], [1, 2, 1 / 4]],
        )
        assert_array_almost_equal(expected, result.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1]]), result.metas)

        # data 2
        self.assertEqual(data.domain, result2.domain)
        attrs = result2.domain.attributes
        self.assertIsInstance(attrs[0], ContinuousVariable)
        self.assertIsInstance(attrs[1], DiscreteVariable)
        self.assertIsInstance(attrs[2], ContinuousVariable)
        self.assertEqual("foo", attrs[0].name)
        self.assertEqual("b - Mean", attrs[1].name)
        self.assertEqual("b - Sum", attrs[2].name)
        self.assertTupleEqual(("0.0", "0.6000000000000001", "1.0"), attrs[1].values)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result2.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result2.domain.metas[0].values)

        expected = np.array(
            [
                [0, np.nan, -1 / 4],
                [0.33, 0, 1 / 4],
                [0.67, 2, 1 / 4],
                [1, 0, 1 / 4],
                [1.33, np.nan, -2 / 4],
            ],
        )
        assert_array_almost_equal(expected, result2.X, decimal=2)
        assert_array_equal(np.array([[0], [0], [1], [1], [1]]), result2.metas)

    def test_double_slicer(self):
        data = self.apply_init(self.data)
        data = self.apply_slicer(data, "a", [((0, 2.5), "S1"), ((2.5, 10), "S2")])
        data = self.apply_slicer(data, "b", [((0, 3.1), "S11"), ((4.2, 10), "S22")])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        self.assertTupleEqual(
            (DiscreteVariable("Slice"), DiscreteVariable("Slice (1)")),
            result.domain.metas,
        )
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)
        self.assertTupleEqual(("S11", "S22"), result.domain.metas[1].values)

        assert_array_equal(data.X, result.X)
        expected = np.array([[1, 2], [2, 3], [1, 3], [2, 5], [4, 5]])
        assert_array_equal(expected, result.X)
        assert_array_equal(
            np.array([[0, 0], [0, 0], [0, 0], [0, 1], [1, 1]]), result.metas
        )

        # data 2
        self.assertEqual(5, len(result2.domain))
        self.assertTupleEqual(
            (DiscreteVariable("Slice"), DiscreteVariable("Slice (1)")),
            result2.domain.metas,
        )
        self.assertTupleEqual(("S1", "S2"), result2.domain.metas[0].values)
        self.assertTupleEqual(("S11", "S22"), result2.domain.metas[1].values)

        expected = np.array(
            [[2, 2], [1, 1], [1, 2], [2, 3], [5, 2], [4, 3], [4, 2], [3, 5]]
        )
        assert_array_equal(expected, result2.X)
        assert_array_equal(
            np.array([[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0], [1, 1]]),
            result2.metas,
        )

    def test_double_aggregate(self):
        data = self.apply_init(self.data)
        data = self.apply_aggregation(data, [1], [2])
        data = self.apply_aggregation(data, [1], [0, 2])
        trans = data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertListEqual(["a"], [a.name for a in trans[1].row_attrs])
        self.assertListEqual(["b"], [a.name for a in trans[1].val_attrs])
        self.assertListEqual(["b - Mean"], [a.name for a in trans[2].row_attrs])
        self.assertListEqual(["a", "b - Sum"], [a.name for a in trans[2].val_attrs])

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        expected_domain = (
            ContinuousVariable("b - Mean"),
            ContinuousVariable("a - Mean"),
            ContinuousVariable("a - Sum"),
            ContinuousVariable("b - Sum - Mean"),
            ContinuousVariable("b - Sum - Sum"),
        )
        self.assertTupleEqual(expected_domain, result.domain.attributes)

        assert_array_equal(data.X, result.X)
        expected = np.array([[2.5, 1, 1, 5, 5], [4, 2.5, 5, 6, 12], [5, 4, 4, 5, 5]])
        assert_array_equal(expected, result.X)

        # data 2
        self.assertEqual(data.domain, result2.domain)
        self.assertTupleEqual(expected_domain, result2.domain.attributes)
        expected = np.array(
            [[1.5, 1, 1, 3, 3], [2, 5, 5, 2, 2], [2.5, 3, 6, 5, 10], [5, 3, 3, 5, 5]]
        )
        assert_array_equal(expected, result2.X)

    def test_exception(self):
        iris = Table("iris")
        data = self.apply_init(self.data)
        data = self.apply_slicer(data, "a", [((0, 2.1), "S1"), ((2.5, 10), "S2")])

        with random_temp_file() as path:
            self.apply_export(data, path)
            with self.assertRaises(ValueError) as err:
                self.apply_transformation(path, iris)
        self.assertEqual(
            "The data are missing the following features: a, b, class",
            str(err.exception),
        )

    def test_double_transformation(self):
        """
        Domain from slicer contain foo variable that has a -> foo compute_value
        in domain - with adding another Orange transformation we check that
        it doesn't get applied twice when evaluated on new data
        """
        data = self.apply_init(self.data)
        data = self.apply_edit_domain(data, ((0, Rename("foo")),))
        data = self.apply_slicer(data, "foo", [((0, 2.5), "S1"), ((2.5, 10), "S2")])

        data = self.apply_edit_domain(data, ((1, Rename("bar")),))

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)

        attrs = result.domain.attributes
        self.assertIsInstance(attrs[0], ContinuousVariable)
        self.assertIsInstance(attrs[1], ContinuousVariable)
        self.assertEqual("foo", attrs[0].name)
        self.assertEqual("bar", attrs[1].name)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)
        assert_array_equal(data.X, result.X)
        expected = np.array([[1, 2], [2, 3], [1, 3], [2, 5], [3, 4], [4, 5]])
        assert_array_equal(expected, result.X)
        assert_array_equal(np.array([[0], [0], [0], [0], [1], [1]]), result.metas)

    def test_stepwise(self):
        data = self.apply_init(self.data)
        data = self.apply_stepwise_feature_selection(data)

        domain2 = Domain(self.data2.domain.attributes, [ContinuousVariable("class")])
        data2 = self.data2.transform(domain2)
        with data2.unlocked(data2.Y):
            data2[:, "class"] = np.arange(0, len(data2))

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, data2)

        self.assertEqual(data.domain, result.domain)
        self.assertTupleEqual((ContinuousVariable("a"),), result.domain.attributes)
        self.assertTupleEqual((ContinuousVariable("class"),), result.domain.class_vars)

        assert_array_equal(data.X, result.X)
        assert_array_equal(np.array([[1], [2], [3], [4], [1], [2]]), result.X)
        assert_array_equal(np.array([1, 2, 3, 4, 1, 2]), result.Y)
        assert_array_equal(np.empty((6, 0)), result.metas)

        self.assertEqual(data.domain, result2.domain)
        self.assertTupleEqual((ContinuousVariable("a"),), result2.domain.attributes)
        self.assertTupleEqual((ContinuousVariable("class"),), result2.domain.class_vars)
        assert_array_equal(
            np.array([[2], [1], [1], [3], [2], [5], [4], [4]]), result2.X
        )
        assert_array_equal(np.arange(0, 8), result2.Y)
        assert_array_equal(np.empty((8, 0)), result2.metas)

    def test_stepwise_with_orange_transform(self):
        data = self.apply_init(self.data)
        data = self.apply_normalize(data)
        data = self.apply_stepwise_feature_selection(data)

        domain2 = Domain(self.data2.domain.attributes, [ContinuousVariable("class")])
        data2 = self.data2.transform(domain2)
        with data2.unlocked(data2.Y):
            data2[:, "class"] = np.arange(0, len(data2))

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, data2)

        self.assertEqual(data.domain, result.domain)
        self.assertEqual(1, len(result.domain.attributes))
        self.assertEqual("a", result.domain.attributes[0].name)
        self.assertTupleEqual((ContinuousVariable("class"),), result.domain.class_vars)

        assert_array_equal(data.X, result.X)
        assert_array_almost_equal(
            np.array([[0], [0.33], [0.66], [1], [0], [0.33]]), result.X, decimal=2
        )
        assert_array_equal(np.array([1, 2, 3, 4, 1, 2]), result.Y)
        assert_array_equal(np.empty((6, 0)), result.metas)

        self.assertEqual(data.domain, result2.domain)
        self.assertEqual(1, len(result.domain.attributes))
        self.assertEqual("a", result.domain.attributes[0].name)
        self.assertTupleEqual((ContinuousVariable("class"),), result2.domain.class_vars)

        expected = np.array([[0.33], [0], [0], [0.66], [0.33], [1.33], [1], [1]])
        assert_array_almost_equal(expected, result2.X, decimal=2)
        assert_array_equal(np.arange(0, 8), result2.Y)
        assert_array_equal(np.empty((8, 0)), result2.metas)

    def test_stepwise_with_orange_transform2(self):
        data = self.apply_init(self.data)
        data = self.apply_stepwise_feature_selection(data)
        data = self.apply_normalize(data)

        with random_temp_file() as path:
            self.apply_export(data, path)
            result = self.apply_transformation(path, self.data)
            result2 = self.apply_transformation(path, self.data2)

        self.assertEqual(data.domain, result.domain)
        self.assertEqual(1, len(result.domain.attributes))
        self.assertEqual("a", result.domain.attributes[0].name)
        self.assertTupleEqual((ContinuousVariable("class"),), result.domain.class_vars)

        assert_array_equal(data.X, result.X)
        assert_array_almost_equal(
            np.array([[0], [0.33], [0.66], [1], [0], [0.33]]), result.X, decimal=2
        )
        assert_array_equal(np.array([1, 2, 3, 4, 1, 2]), result.Y)
        assert_array_equal(np.empty((6, 0)), result.metas)

        self.assertEqual(data.domain, result2.domain)
        self.assertEqual(1, len(result.domain.attributes))
        self.assertEqual("a", result.domain.attributes[0].name)
        self.assertTupleEqual((ContinuousVariable("class"),), result2.domain.class_vars)

        expected = np.array([[0.33], [0], [0], [0.66], [0.33], [1.33], [1], [1]])
        assert_array_almost_equal(expected, result2.X, decimal=2)
        assert_array_equal(np.arange(0, 8), result2.Y)
        assert_array_equal(np.empty((8, 0)), result2.metas)


if __name__ == "__main__":
    unittest.main()
