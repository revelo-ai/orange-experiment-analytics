import unittest
from functools import partial
from typing import List

import numpy as np
import pandas as pd
from AnyQt.QtCore import QItemSelectionModel
from AnyQt.QtWidgets import QListView
from Orange.data import (
    Table,
    table_to_frame,
    ContinuousVariable,
    Domain,
    DiscreteVariable,
    TimeVariable,
)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    InfoTransform,
    HTML_TABLE_STYLE,
)
from orangecontrib.experiment_analytics.widgets.owaggregate import (
    OWAggregate,
    AggregatePreprocessor,
    auc,
    abs_auc,
    polynomial_function,
    custom_function,
)


def create_sample_data():
    domain = Domain(
        [
            ContinuousVariable("a"),
            ContinuousVariable("b"),
            ContinuousVariable("rid"),
            ContinuousVariable("cval"),
        ]
    )
    return Table.from_numpy(
        domain,
        np.array(
            [
                [1, 1, 1, 0.1],
                [1, 1, 2, 0.2],
                [1, 2, 1, 0.2],
                [1, 2, 2, 0.3],
                [1, 3, 1, 0.3],
                [1, 3, 2, 0.4],
                [1, 3, 3, 0.6],
                [2, 1, 1, 1.0],
                [2, 1, 2, 2.0],
                [2, 2, 1, 3.0],
                [2, 2, 2, -4.0],
                [2, 3, 1, 5.0],
                [2, 3, 2, 5.0],
            ]
        ),
    )


def create_frequency_data():
    domain = Domain(
        [
            ContinuousVariable("group"),
            ContinuousVariable("x"),
            ContinuousVariable("y"),
        ]
    )
    x = np.linspace(0, 10 * np.pi, 1000).reshape((-1, 1))
    y1 = np.sin(x)  # period 2pi and amplitude 1
    y2 = np.sin(x * 2) * 2  # period pi and amplitude 2
    return Table.from_numpy(
        domain,
        np.vstack(
            (
                np.hstack((np.ones(x.shape), x, y1)),
                np.hstack((np.ones(x.shape) * 2, x, y2)),
            )
        ),
    )


class TestOWAggregate(WidgetTest):
    def setUp(self) -> None:
        self.widget = self.create_widget(OWAggregate)
        self.data = Table("heart_disease")
        self.small_data = create_sample_data()

    def test_input(self):
        self.send_signal(self.widget.Inputs.data, self.data)

        domain = self.data.domain
        self.assertListEqual(self.widget.row_attrs, [domain.class_var])
        self.assertListEqual(self.widget.value_attrs, [domain.attributes[0]])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertEqual(self.widget.aggregations, {"Mean"})

        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output.domain.attributes), 2)

    @staticmethod
    def _set_selection(view: QListView, indices: List[int]):
        sm = view.selectionModel()
        model = view.model()
        for ind in indices:
            sm.select(model.index(ind), QItemSelectionModel.Select)

    def test_rows(self):
        domain = self.data.domain
        self.send_signal(self.widget.Inputs.data, self.data)

        # test default row selection - diameter narrowing
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 2)
        self.assertListEqual(
            list(map(str, output.domain.attributes)),
            ["diameter narrowing", "age - Mean"]
        )
        self.assertTupleEqual(
            output.domain["diameter narrowing"].values, ("0", "1")
        )

        # select diameter narrowing and age
        self._set_selection(self.widget.row_attrs_view, [0, 1])
        self.assertListEqual(
            self.widget.row_attrs, [domain["diameter narrowing"], domain["age"]]
        )
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        ages = np.unique(self.data.get_column("age"))

        self.assertListEqual(
            list(map(str, output.domain.attributes)),
            ["diameter narrowing", "age", "age - Mean"],
        )
        # age is ContinuousVariable
        self.assertTupleEqual(
            output.domain["diameter narrowing"].values, ("0", "1")
        )
        self.assertSetEqual(
            set(output.get_column_view("age")[0].tolist()), set(ages)
        )
        self.assertListEqual(
            table_to_frame(output).iloc[:, :2].values.tolist()[:40],
            [[s, d] for s in ("0", "1") for d in ages][:40],
        )

    def test_columns(self):
        domain = self.data.domain
        self.send_signal(self.widget.Inputs.data, self.data)

        # test one column selected
        self._set_selection(self.widget.col_attrs_view, [1])
        self.assertListEqual(self.widget.row_attrs,
                             [domain["diameter narrowing"]])
        self.assertListEqual(self.widget.col_attrs, [domain["age"]])
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        ages = np.unique(self.data.get_column("age")).astype(int)

        self.assertListEqual(
            list(map(str, output.domain.attributes)),
            ["diameter narrowing"] + [f"age - {a} - Mean" for a in ages],
        )
        self.assertEqual(len(output), 2)

        # two columns - select age and gender
        self._set_selection(self.widget.col_attrs_view, [1, 2])
        self.assertListEqual(self.widget.row_attrs,
                             [domain["diameter narrowing"]])
        self.assertListEqual(
            self.widget.col_attrs, [domain["age"], domain["gender"]]
        )
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertListEqual(
            list(map(str, output.domain.attributes)),
            ["diameter narrowing"] + [f"age - {a} - {g} - Mean" for a in ages
                                      for g in ("female", "male")],
        )
        self.assertEqual(len(output), 2)

    def test_values(self):
        domain = self.data.domain
        self.send_signal(self.widget.Inputs.data, self.data)

        # test one column selected
        self.widget.val_attrs_view.clearSelection()
        self._set_selection(self.widget.val_attrs_view, [1])
        self.assertListEqual(self.widget.row_attrs, [domain["diameter narrowing"]])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertListEqual(self.widget.value_attrs, [domain["rest SBP"]])
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertListEqual(
            list(map(str, output.domain.attributes)),
            ["diameter narrowing", "rest SBP - Mean"],
        )
        self.assertEqual(len(output), 2)

        # test with discrete attribute
        self._set_selection(self.widget.val_attrs_view, [2])
        self.assertListEqual(self.widget.row_attrs, [domain["diameter narrowing"]])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertListEqual(
            self.widget.value_attrs, [domain["rest SBP"], domain["cholesterol"]]
        )
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertListEqual(
            list(map(str, output.domain.attributes)),
            ["diameter narrowing", "rest SBP - Mean", "cholesterol - Mean"],
        )
        self.assertEqual(len(output), 2)

    def _get_checked(self):
        return [
            agg for agg, cb in self.widget.aggregation_cbs.items() if cb.isChecked()
        ]

    def test_context(self):
        domain = self.data.domain
        self.send_signal(self.widget.Inputs.data, self.data)

        r_view = self.widget.row_attrs_view
        c_view = self.widget.col_attrs_view
        v_view = self.widget.val_attrs_view

        self.widget.aggregation_cbs["Count"].setChecked(True)
        self._set_selection(self.widget.row_attrs_view, [1])
        self.assertListEqual(
            self.widget.row_attrs, [domain["diameter narrowing"], domain["age"]]
        )
        self.assertListEqual(self.widget.value_attrs, [domain["age"]])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertEqual(self.widget.aggregations, {"Mean", "Count"})

        self.send_signal(self.widget.Inputs.data, None)
        self.assertListEqual(self.widget.row_attrs, [])
        self.assertListEqual(self.widget.value_attrs, [])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertEqual(self.widget.aggregations, {"Mean"})

        # test if variables set back correctly
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertListEqual(
            self.widget.row_attrs, [domain["diameter narrowing"], domain["age"]]
        )
        self.assertListEqual(
            self.widget._get_selection(r_view), [domain["diameter narrowing"], domain["age"]]
        )
        self.assertListEqual(self.widget.value_attrs, [domain["age"]])
        self.assertListEqual(self.widget._get_selection(v_view), [domain["age"]])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertListEqual(self.widget._get_selection(c_view), [])
        self.assertEqual(self.widget.aggregations, {"Mean", "Count"})
        self.assertListEqual(self._get_checked(), ["Mean", "Count"])

        self.send_signal(self.widget.Inputs.data, None)
        self.assertListEqual(self.widget.row_attrs, [])
        self.assertListEqual(self.widget.value_attrs, [])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertEqual(self.widget.aggregations, {"Mean"})

        # context should correctly match the table with an additinal attribute
        d = self.data.add_column(ContinuousVariable("a"), np.ones(len(self.data)))
        self.send_signal(self.widget.Inputs.data, d)
        self.assertListEqual(
            self.widget.row_attrs, [domain["diameter narrowing"], domain["age"]]
        )
        self.assertListEqual(
            self.widget._get_selection(r_view), [domain["diameter narrowing"], domain["age"]]
        )
        self.assertListEqual(self.widget.value_attrs, [domain["age"]])
        self.assertListEqual(self.widget._get_selection(v_view), [domain["age"]])
        self.assertListEqual(self.widget.col_attrs, [])
        self.assertListEqual(self.widget._get_selection(c_view), [])
        self.assertEqual(self.widget.aggregations, {"Mean", "Count"})
        self.assertListEqual(self._get_checked(), ["Mean", "Count"])

    def test_exception(self):
        self.send_signal(self.widget.Inputs.data, self.data)

        # input incorrect custom function which will raise error
        self.widget.aggregation_cbs["Custom function"].setChecked(True)
        self.widget.controls.custom_function_name.setText("fun")
        self.widget.controls.custom_function.setText("bar")
        self.widget.commit.now()
        self.wait_until_finished()

        # test error raised
        self.assertTrue(self.widget.Warning.cannot_compute.is_shown())
        self.assertEqual(
            str(self.widget.Warning.cannot_compute),
            "Some scores cannot be computed: name 'bar' is not defined",
        )

        # unselect custom function to check if error disappears
        self.widget.aggregation_cbs["Custom function"].setChecked(False)
        # cause same row - column error
        self._set_selection(self.widget.col_attrs_view, [0])

        # check if custom function error disappeared and same row - column error appeared
        self.assertFalse(self.widget.Warning.cannot_compute.is_shown())
        self.assertTrue(self.widget.Error.row_col_intersection.is_shown())

        # remove error cause an check if disappeared
        self.widget.col_attrs_view.clearSelection()
        self.assertFalse(self.widget.Error.row_col_intersection.is_shown())

    def test_aggregations(self):
        self.send_signal(self.widget.Inputs.data, self.small_data)
        self._set_selection(self.widget.row_attrs_view, [0, 1])  # a, b
        self.widget.val_attrs_view.clearSelection()
        self._set_selection(self.widget.val_attrs_view, [3])  # cval
        self.widget.controls.custom_function.setText("mean(y)")
        simulate.combobox_activate_item(self.widget.controls.x_variable, "rid")

        for cb in self.widget.aggregation_cbs.values():
            cb.setChecked(True)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        # fmt: off
        res = [
            [1, 1, 0.15, 0.15, 0.070, 0.005, 0.3, 0.1, 0.2, 2, 2, 0.15, 0.15,
             0.1, 0, 0.0214286, 0.0357143, 0.0428571, 0.2, 0, 0.15],
            [1, 2, 0.25, 0.25, 0.070, 0.005, 0.5, 0.2, 0.3, 2, 2, 0.25, 0.25,
             0.1, 0.1, 0.0157143, 0.0528571, 0.131429, 0.2, 0, 0.25],
            [1, 3, 0.433, 0.4, 0.152, 0.023, 1.3, 0.3, 0.6, 3, 3, 0.85, 0.85,
             0.15, 0.133333, 0.05, -0.05, 0.3, 0.1, 0, 0.433],
            [2, 1, 1.5, 1.5, 0.707, 0.5, 3, 1, 2, 2, 2, 1.5, 1.5, 1, 0,
             0.214286, 0.357143, 0.428571, 0.2, 0, 1.5],
            [2, 2, -0.5, -0.5, 4.949, 24.5, -1, -4, 3, 2, 2, -0.5, 3.5, -7, 10,
             -2.07143, -0.785714, 5.85714, 0.2, 0, -0.5],
            [2, 3, 5, 5, 0, 0, 10, 5, 5, 2, 2, 5, 5, 0, 5, -0.285714, 0.857143,
             4.42857, 0.2, 0, 5],
        ]
        # fmt: on
        np.testing.assert_array_almost_equal(
            table_to_frame(output).values, res, decimal=3
        )

        # select just three otherwise table will be too wide
        for cb in self.widget.aggregation_cbs.values():
            cb.setChecked(False)
        for cb in list(self.widget.aggregation_cbs.values())[:3]:
            cb.setChecked(True)

        self.widget.row_attrs_view.clearSelection()
        self._set_selection(self.widget.row_attrs_view, [0])  # a
        self._set_selection(self.widget.col_attrs_view, [1])  # b
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        res = [
            [1, 0.15, 0.25, 0.433, 0.15, 0.25, 0.4, 0.071, 0.071, 0.153],
            [2, 1.5, -0.5, 5, 1.5, -0.5, 5, 0.707, 4.95, 0],
        ]
        np.testing.assert_array_almost_equal(
            table_to_frame(output).values, res, decimal=3
        )

    def test_metas(self):
        """
        Test if columns that were in metas in original data are also in metas
        in new data
        """
        self.send_signal(self.widget.Inputs.data, self.data)
        self._set_selection(self.widget.row_attrs_view, [0, 1])
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertListEqual(
            ["diameter narrowing", "age", "age - Mean"],
            [v.name for v in output.domain.attributes],
        )
        self.assertEqual(0, len(output.domain.metas))
        self.assertIsInstance(output.domain["diameter narrowing"], DiscreteVariable)
        self.assertIsInstance(output.domain["age"], ContinuousVariable)

        # move one attribute to metas
        d = self.data.domain
        new_domain = Domain(
            [v for v in d.attributes if v.name != "diameter narrowing"], metas=[d["diameter narrowing"]]
        )
        data = self.data.from_table(new_domain, self.data)
        self.send_signal(self.widget.Inputs.data, data)
        self._set_selection(self.widget.row_attrs_view, [0, 1])
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertListEqual(
            ["age", "age - Mean"], [v.name for v in output.domain.attributes]
        )
        self.assertListEqual(["diameter narrowing"], [v.name for v in output.domain.metas])
        self.assertIsInstance(output.domain["diameter narrowing"], DiscreteVariable)
        self.assertIsInstance(output.domain["age"], ContinuousVariable)

        # move two attributes to metas
        new_domain = Domain(
            [v for v in d.attributes if v.name not in ("diameter narrowing", "age")],
            metas=[d["diameter narrowing"], d["age"]],
        )
        data = self.data.from_table(new_domain, self.data)
        self.send_signal(self.widget.Inputs.data, data)
        self._set_selection(self.widget.row_attrs_view, [0, 1])
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)

        self.assertListEqual(
            ["age - Mean"], [v.name for v in output.domain.attributes]
        )
        self.assertListEqual(
            ["diameter narrowing", "age"], [v.name for v in output.domain.metas]
        )
        self.assertIsInstance(output.domain["diameter narrowing"], DiscreteVariable)
        self.assertIsInstance(output.domain["age"], ContinuousVariable)

    def test_types(self):
        """
        Test whether type of row attributes is kept
        This test tests only Time variable since all other types are tested
        in previous method
        """
        new_tab = Table.from_numpy(
            Domain(self.small_data.domain.attributes, metas=[TimeVariable("tvar")]),
            self.small_data.X,
            metas=self.small_data.X[:, 0:1],
        )
        self.send_signal(self.widget.Inputs.data, new_tab)
        self._set_selection(self.widget.row_attrs_view, [4])

        output = self.get_output(self.widget.Outputs.data)
        self.assertIsInstance(output.domain["tvar"], TimeVariable)

    def test_frequency(self):
        data = create_frequency_data()
        self.send_signal(self.widget.Inputs.data, data)
        self.widget.row_attrs_view.clearSelection()
        self._set_selection(self.widget.row_attrs_view, [0])  # group
        self._set_selection(self.widget.val_attrs_view, [2])  # y
        simulate.combobox_activate_item(self.widget.controls.x_variable, "x")

        for cb in self.widget.aggregation_cbs.values():
            cb.setChecked(False)
        self.widget.aggregation_cbs["Frequency"].setChecked(True)
        output = self.get_output(self.widget.Outputs.data)
        res = [[1, 0.159, 0.975], [2, 0.318, 1.987]]
        np.testing.assert_array_almost_equal(output.X, res, decimal=2)
        self.assertListEqual(
            ["group", "y - Frequency", "y - Amplitude"],
            [a.name for a in output.domain.attributes],
        )

        self.widget.controls.freq_r2.setChecked(True)
        output = self.get_output(self.widget.Outputs.data)
        res = [[1, 0.159, 0.975, 0.98], [2, 0.318, 1.987, 0.99]]
        np.testing.assert_array_almost_equal(output.X, res, decimal=2)
        self.assertListEqual(
            ["group", "y - Frequency", "y - Amplitude", "y - R2"],
            [a.name for a in output.domain.attributes],
        )

        self.widget.controls.freq_damping.setChecked(True)
        output = self.get_output(self.widget.Outputs.data)
        # half-life very big negative number since there is no damping
        self.assertLess(output.X[0, 3], -1e-5)
        self.assertLess(output.X[1, 3], -1e-5)
        self.assertListEqual(
            ["group", "y - Frequency", "y - Amplitude", "y - Half-life", "y - R2"],
            [a.name for a in output.domain.attributes],
        )

    def test_custom_function(self):
        self.send_signal(self.widget.Inputs.data, self.small_data)
        # select just three otherwise table will be too wide
        for cb in self.widget.aggregation_cbs.values():
            cb.setChecked(False)

        self.widget.row_attrs_view.clearSelection()
        self._set_selection(self.widget.row_attrs_view, [0])  # a
        self.widget.controls.custom_function.setText("mean(y)")
        self.widget.aggregation_cbs["Custom function"].setChecked(True)

        output = self.get_output(self.widget.Outputs.data)
        self.assertListEqual(
            ["a", "b - Custom function"], [a.name for a in output.domain.attributes]
        )

        self.widget.controls.custom_function.setText("frequency(x, y)")
        self.widget.commit.now()
        output = self.get_output(self.widget.Outputs.data)
        self.assertListEqual(
            ["a", "b - Frequency", "b - Amplitude"],
            [a.name for a in output.domain.attributes],
        )

        self.widget.controls.custom_function.setText("frequency(x, y, True)")
        self.widget.commit.now()
        output = self.get_output(self.widget.Outputs.data)
        self.assertListEqual(
            ["a", "b - Frequency", "b - Amplitude", "b - Half-life"],
            [a.name for a in output.domain.attributes],
        )

        self.widget.controls.custom_function.setText("frequency(x, y, True, True)")
        self.widget.commit.now()
        output = self.get_output(self.widget.Outputs.data)
        self.assertListEqual(
            ["a", "b - Frequency", "b - Amplitude", "b - Half-life", "b - R2"],
            [a.name for a in output.domain.attributes],
        )

    def test_sub_control_enabling(self):
        self.send_signal(self.widget.Inputs.data, self.small_data)
        self.assertFalse(self.widget.controls.freq_damping.isEnabled())
        self.widget.aggregation_cbs["Frequency"].click()
        self.assertTrue(self.widget.controls.freq_damping.isEnabled())

    def test_transformation_added(self):
        # table doesn't have transformations - attribute no transformation is added
        self.send_signal(self.widget.Inputs.data, self.small_data)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertNotIn(TRANSFORMATIONS_ATTRIBUTE, output.attributes)

        # table has transformations - add domain and aggregation preprocessing
        ti = InfoTransform(self.small_data.domain)
        self.small_data.attributes[TRANSFORMATIONS_ATTRIBUTE] = (ti,)
        self.send_signal(self.widget.Inputs.data, self.small_data)
        self._set_selection(self.widget.row_attrs_view, [0])  # a
        self._set_selection(self.widget.col_attrs_view, [1])  # b
        self.widget.val_attrs_view.clearSelection()
        self._set_selection(self.widget.val_attrs_view, [3])  # cval
        self.widget.controls.custom_function.setText("mean(y)")
        simulate.combobox_activate_item(self.widget.controls.x_variable, "rid")
        self.widget.aggregation_cbs["Sum"].click()
        self.widget.aggregation_cbs["Custom function"].click()
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        transformations = output.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(transformations))

        # test info transformation - always in the beginning of workflow
        self.assertIsInstance(transformations[0], InfoTransform)
        self.assertEqual(self.small_data.domain, transformations[0].domain)

        # test aggregation transformation
        d = self.small_data.domain
        self.assertIsInstance(transformations[1], AggregatePreprocessor)
        self.assertListEqual([d["a"]], transformations[1].row_attrs)
        self.assertListEqual([d["b"]], transformations[1].col_attrs)
        self.assertListEqual([d["cval"]], transformations[1].val_attrs)
        self.assertEqual(d["rid"], transformations[1].x_attr)
        expected = [("Mean", "mean"), ("Sum", "sum")]
        self.assertListEqual(expected, transformations[1].aggregations[:2])
        expected = ("Custom function", "Custom function", "mean(y)")
        self.assertEqual(expected, transformations[1].aggregations[2][0])
        self.assertIsInstance(transformations[1].aggregations[2][1], partial)


class TestAggregatePreprocessor(unittest.TestCase):
    def setUp(self):
        self.small_data = create_sample_data()

    def test_aggregations(self):
        d = self.small_data.domain
        agg = [
            ("Mean", "mean"),
            ("Median", "median"),
            ("Standard deviation", "std"),
            ("Variance", "var"),
            ("Sum", "sum"),
            ("Min", "min"),
            ("Max", "max"),
            ("Count defined", "count"),
            ("Count", "size"),
            ("Area under the curve", auc),
            ("Absolute area under the curve", abs_auc),
            ("Linear fit", partial(polynomial_function, 1)),
            ("Quadratic fit", partial(polynomial_function, 2)),
            (
                ("Custom function", "Custom function", "mean(y)"),
                partial(custom_function, "Custom function", "mean(y)"),
            ),
        ]
        ap = AggregatePreprocessor([d["a"], d["b"]], [], [d["cval"]], agg, d["rid"])
        result = ap(self.small_data)
        # fmt: off
        expected = np.array([
            [1, 1, 0.15, 0.15, 0.070, 0.005, 0.3, 0.1, 0.2, 2, 2, 0.15, 0.15,
             0.1, 0, 0.0214286, 0.0357143, 0.0428571, 0.15],
            [1, 2, 0.25, 0.25, 0.070, 0.005, 0.5, 0.2, 0.3, 2, 2, 0.25, 0.25,
             0.1, 0.1, 0.0157143, 0.0528571, 0.131429, 0.25],
            [1, 3, 0.433, 0.4, 0.152, 0.023, 1.3, 0.3, 0.6, 3, 3, 0.85, 0.85,
             0.15, 0.133333, 0.05, -0.05, 0.3, 0.433],
            [2, 1, 1.5, 1.5, 0.707, 0.5, 3, 1, 2, 2, 2, 1.5, 1.5, 1, 0,
             0.214286, 0.357143, 0.428571, 1.5],
            [2, 2, -0.5, -0.5, 4.949, 24.5, -1, -4, 3, 2, 2, -0.5, 3.5, -7, 10,
             -2.07143, -0.785714, 5.85714, -0.5],
            [2, 3, 5, 5, 0, 0, 10, 5, 5, 2, 2, 5, 5, 0, 5, -0.285714, 0.857143,
             4.42857, 5],
        ])
        # fmt: on
        np.testing.assert_array_almost_equal(
            table_to_frame(result).values, expected, decimal=3
        )

        agg = [("Mean", "mean"), ("Median", "median"), ("Standard deviation", "std")]
        ap = AggregatePreprocessor([d["a"]], [d["b"]], [d["cval"]], agg, d["rid"])
        result = ap(self.small_data)
        expected = np.array(
            [
                [1, 0.15, 0.25, 0.433, 0.15, 0.25, 0.4, 0.071, 0.071, 0.153],
                [2, 1.5, -0.5, 5, 1.5, -0.5, 5, 0.707, 4.95, 0],
            ]
        )
        np.testing.assert_array_almost_equal(
            table_to_frame(result).values, expected, decimal=3
        )

    def test_missing_attribute(self):
        d = self.small_data.domain
        missing = ContinuousVariable("Missing")
        missing1 = ContinuousVariable("Missing 1")
        agg = [("Mean", "mean"), ("Median", "median"), ("Standard deviation", "std")]

        ap = AggregatePreprocessor(
            [missing, missing1, d["a"]], [d["b"]], [d["cval"]], agg, d["rid"]
        )
        with self.assertRaises(ValueError) as err:
            ap(self.small_data)
        self.assertEqual(
            "Data missing attributes: Missing, Missing 1", str(err.exception)
        )

        ap = AggregatePreprocessor(
            [d["a"]], [d["b"], missing, missing1], [d["cval"]], agg, d["rid"]
        )
        with self.assertRaises(ValueError) as err:
            ap(self.small_data)
        self.assertEqual(
            "Data missing attributes: Missing, Missing 1", str(err.exception)
        )

        agg = [("Mean", "mean"), ("Median", "median"), ("Standard deviation", "std")]
        ap = AggregatePreprocessor([d["a"]], [d["b"]], [missing1], agg, missing)
        with self.assertRaises(ValueError) as err:
            ap(self.small_data)
        self.assertEqual(
            "Data missing attributes: Missing, Missing 1", str(err.exception)
        )

    def test_repr(self):
        d = self.small_data.domain
        agg = [("Mean", "mean"), ("Median", "median"), ("Standard deviation", "std")]
        ap = AggregatePreprocessor([d["a"]], [d["b"]], [d["cval"]], agg, d["rid"])
        expected = (
            "<h4>Aggregate</h4>"
            f"{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Rows (Group by): </th><td>a</td></tr>"
            "<tr><th>Columns (Split by): </th><td>b</td></tr>"
            "<tr><th>Values to aggregate: </th><td>cval</td></tr>"
            "<tr><th>Aggregations: </th><td>Mean, Median, Standard deviation</td></tr>"
            "<tr><th>X variable: </th><td>rid</td></tr></table></div>"
        )
        self.assertEqual(expected, str(ap))

        ap = AggregatePreprocessor([d["a"]], [], [d["cval"]], agg, None)
        expected = (
            "<h4>Aggregate</h4>"
            f"{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Rows (Group by): </th><td>a</td></tr>"
            "<tr><th>Columns (Split by): </th><td>N/A</td></tr>"
            "<tr><th>Values to aggregate: </th><td>cval</td></tr>"
            "<tr><th>Aggregations: </th><td>Mean, Median, Standard deviation</td></tr>"
            "<tr><th>X variable: </th><td>N/A</td></tr></table></div>"
        )
        self.assertEqual(expected, str(ap))

        agg = [
            ("Mean", "mean"),
            ("Median", "median"),
            ("Standard deviation", "std"),
            ("Variance", "var"),
            ("Sum", "sum"),
            ("Min", "min"),
            ("Max", "max"),
            ("Count defined", "count"),
            ("Count", "size"),
            ("Area under the curve", auc),
            ("Linear fit", partial(polynomial_function, 1)),
            ("Quadratic fit", partial(polynomial_function, 2)),
            (
                ("Custom function", "Custom function", "mean(y)"),
                partial(custom_function, "Custom function", "mean(y)"),
            ),
        ]
        ap = AggregatePreprocessor([d["a"], d["b"]], [], [d["cval"]], agg, d["rid"])
        expected = (
            "<h4>Aggregate</h4>"
            f"{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Rows (Group by): </th><td>a, b</td></tr>"
            "<tr><th>Columns (Split by): </th><td>N/A</td></tr>"
            "<tr><th>Values to aggregate: </th><td>cval</td></tr>"
            "<tr><th>Aggregations: </th><td>Mean, Median, Standard deviation, "
            "Variance, Sum, Min, Max, Count defined, Count, Area under the curve, "
            "Linear fit, Quadratic fit, Custom function</td></tr>"
            "<tr><th>X variable: </th><td>rid</td></tr></table></div>"
        )
        self.assertEqual(expected, str(ap))

    def test_abs_auc_sorted(self):
        df = pd.DataFrame([[0, 0],
                           [2, 2],
                           [4, 0],
                           [1, 1],
                           [3, 1]])
        self.assertEqual(abs_auc(df), 4)


if __name__ == "__main__":
    unittest.main()
