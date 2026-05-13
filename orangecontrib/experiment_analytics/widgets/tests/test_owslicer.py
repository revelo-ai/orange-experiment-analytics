import unittest
from unittest.mock import Mock
import os

import numpy as np
from AnyQt.QtCore import QItemSelectionModel, QDateTime, Qt, QItemSelection
from AnyQt.QtGui import QFont, QPen, QColor
from AnyQt.QtWidgets import QLabel, QLineEdit, QPushButton, QDoubleSpinBox, \
    QDateTimeEdit

from pyqtgraph.Point import Point

from Orange.data import (
    Table,
    ContinuousVariable,
    TimeVariable,
    FilterContinuous,
    Values,
    Domain,
    DiscreteVariable,
)
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    InfoTransform,
    HTML_TABLE_STYLE,
)
from orangecontrib.experiment_analytics.widgets.owslicer import (
    OWSlicer,
    SlicePicker,
    NonUniqueSeries,
    Runner,
    SlicerPreprocessor,
)


class TestRunner(unittest.TestCase):
    def setUp(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        path = os.path.join(os.path.normpath(path),
                            "datasets", "airpassengers.csv")
        self.data = Table(path)

    def test_valid_series(self):
        x_data = np.arange(3)
        key = np.zeros((3, 1))
        Runner._check_valid_series(x_data, key)

        with self.assertRaises(NonUniqueSeries):
            x_data = np.hstack([np.arange(3), np.arange(3)])
            key = np.zeros((6, 1))
            Runner._check_valid_series(x_data, key)

        key = np.vstack([np.zeros((3, 1)), np.ones((3, 1))])
        Runner._check_valid_series(x_data, key)

        with self.assertRaises(NonUniqueSeries):
            x_data = np.hstack([np.arange(3), np.arange(3),
                                np.arange(3), np.arange(3)])
            key = np.vstack([np.zeros((6, 1)), np.ones((6, 1))])
            Runner._check_valid_series(x_data, key)

        key1 = np.vstack([np.zeros((6, 1)), np.ones((6, 1))])
        key2 = np.vstack([np.zeros((3, 1)), np.ones((3, 1)),
                          np.zeros((3, 1)), np.ones((3, 1))])
        key = np.hstack([key1, key2])
        Runner._check_valid_series(x_data, key)

    def test_run(self):
        key_vars = []
        x_data = self.data.get_column("Month")
        y_data = self.data.get_column("Air passengers")
        keys = [self.data.get_column(var) for var in key_vars]
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)
        result = Runner().run(x_data, y_data, keys, None, None, state)
        self.assertEqual(len(result.groups), 1)
        self.assertIsInstance(result.groups[0].color, QColor)
        self.assertEqual(len(result.groups[0].line), 3)
        self.assertEqual(len(result.groups[0].rnge), 3)
        self.assertEqual(len(result.groups[0].mean), 2)


class TestSlicePicker(WidgetTest):
    def setUp(self):
        self.picker = SlicePicker(None)

    def test_add_row(self):
        self.picker.set_parameters(3, 10, ContinuousVariable("foo"))
        self.picker._add_row()

        controls = self.picker._SlicePicker__controls[0]
        self.assertIsInstance(controls[0], QPushButton)
        self.assertIsInstance(controls[1], QDoubleSpinBox)
        self.assertIsInstance(controls[2], QLabel)
        self.assertIsInstance(controls[3], QDoubleSpinBox)
        self.assertIsInstance(controls[4], QLineEdit)

        self.assertEqual(controls[1].value(), 3)
        self.assertEqual(controls[3].value(), 3)
        self.assertEqual(controls[4].text(), "Slice 1")

        data = self.picker._SlicePicker__data[0]
        self.assertEqual(data[0][0], 3)
        self.assertEqual(data[0][1], 3)
        self.assertEqual(data[1], "Slice 1")

    def test_add_row_with_data(self):
        self.picker.set_parameters(0, 10, ContinuousVariable("foo"))
        self.picker._add_row(2, 4, "Slice")

        controls = self.picker._SlicePicker__controls[0]
        self.assertEqual(controls[1].value(), 2)
        self.assertEqual(controls[3].value(), 4)
        self.assertEqual(controls[4].text(), "Slice")

        data = self.picker._SlicePicker__data[0]
        self.assertEqual(data[0][0], 2)
        self.assertEqual(data[0][1], 4)
        self.assertEqual(data[1], "Slice")

    def test_set_data(self):
        self.picker.set_parameters(0, 10, ContinuousVariable("foo"))
        data = [((-2, 5), "Slice A"), ((1, 11), "Slice B")]
        self.picker.set_data(data)
        self.assertEqual(len(self.picker._SlicePicker__controls), 2)

        controls = self.picker._SlicePicker__controls
        self.assertEqual(controls[0][1].value(), 0)
        self.assertEqual(controls[0][3].value(), 5)
        self.assertEqual(controls[0][4].text(), "Slice A")
        self.assertEqual(controls[1][1].value(), 1)
        self.assertEqual(controls[1][3].value(), 10)
        self.assertEqual(controls[1][4].text(), "Slice B")

        data = self.picker._SlicePicker__data
        self.assertEqual(data[0][0][0], 0)
        self.assertEqual(data[0][0][1], 5)
        self.assertEqual(data[0][1], "Slice A")
        self.assertEqual(data[1][0][0], 1)
        self.assertEqual(data[1][0][1], 10)
        self.assertEqual(data[1][1], "Slice B")

    def test_reset_data(self):
        self.picker.set_parameters(0, 10, ContinuousVariable("foo"))
        self.picker.set_data([((2, 5), "Slice")])
        self.picker.set_data([((3, 6), "Slice")])
        self.assertEqual(len(self.picker._SlicePicker__controls), 1)
        self.assertEqual(len(self.picker._SlicePicker__data), 1)

    def test_clear_all(self):
        self.picker.set_parameters(0, 10, ContinuousVariable("foo"))
        self.picker.set_data([((2, 5), "Slice")])
        self.picker.clear_all()
        self.assertEqual(len(self.picker._SlicePicker__controls), 0)
        self.assertEqual(len(self.picker._SlicePicker__data), 0)

    def test_add_row_datetime(self):
        self.picker.set_parameters(
            0, 10 ** 6, TimeVariable("foo", have_date=True)
        )
        self.picker._add_row()

        controls = self.picker._SlicePicker__controls[0]
        self.assertIsInstance(controls[0], QPushButton)
        self.assertIsInstance(controls[1], QDateTimeEdit)
        self.assertIsInstance(controls[2], QLabel)
        self.assertIsInstance(controls[3], QDateTimeEdit)
        self.assertIsInstance(controls[4], QLineEdit)

        self.assertEqual(controls[1].dateTime(),
                         QDateTime.fromSecsSinceEpoch(0))
        self.assertEqual(controls[3].dateTime(),
                         QDateTime.fromSecsSinceEpoch(0))
        self.assertEqual(controls[4].text(), "Slice 1")

        data = self.picker._SlicePicker__data[0]
        self.assertEqual(data[0][0], 0)
        self.assertEqual(data[0][1], 0)
        self.assertEqual(data[1], "Slice 1")


class TestOWSlicer(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSlicer)
        path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        path = os.path.join(os.path.normpath(path),
                            "datasets", "airpassengers.csv")
        self.data = Table(path)

    def test_default(self):
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        sel_data = self.get_output(self.widget.Outputs.selected_data)
        self.assert_table_equal(sel_data, data)
        self.assertTrue(self.widget.Error.non_unique_series.is_shown())

    def test_no_valid_data(self):
        data = self.data.copy()
        data.X[:, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Error.no_valid_data.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_valid_data.is_shown())

    def test_missing_values_x(self):
        data = self.data.copy()
        data.X[::3, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.__init_widget()
        self.assertTrue(self.widget.Information.hidden_instances.is_shown())
        self.assertFalse(self.widget.Error.non_unique_series.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.hidden_instances.is_shown())

    def test_missing_values_y(self):
        data = self.data.copy()
        data.X[::3, 1] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.__init_widget()
        self.assertTrue(self.widget.Information.hidden_instances.is_shown())
        self.assertFalse(self.widget.Error.non_unique_series.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.hidden_instances.is_shown())

    def test_missing_values_key(self):
        data = self.data.copy()
        data.X[::3, 0] = np.nan
        self.send_signal(self.widget.Inputs.data, data)
        self.__init_widget()
        self.assertTrue(self.widget.Information.hidden_instances.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Information.hidden_instances.is_shown())
        self.assertFalse(self.widget.Error.non_unique_series.is_shown())

    def test_only_discrete_variables(self):
        data = Table("titanic")
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.no_continuous_vars.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.no_continuous_vars.is_shown())

    def test_no_display_option(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.controls.show_range.setChecked(False)
        self.widget.controls.show_mean.setChecked(False)
        self.assertTrue(self.widget.Warning.no_display_option.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.no_display_option.is_shown())

    def test_create_output_table(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()

        self.widget.selection = [((-600000000, -590000000), "Foo"),
                                 ((-500000000, -490000000), "Bar")]
        self.widget.commit.deferred()
        data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIn("Slice", data.domain)
        self.assertTupleEqual(data.domain["Slice"].values, ("Foo", "Bar"))
        self.assertEqual(len(data), 8)
        self.assertTrue(all(data.get_column(self.widget.x_var) <= -490000000))
        self.assertTrue(all(data.get_column(self.widget.x_var) >= -600000000))
        self.assertFalse(self.widget.Error.non_unique_series.is_shown())

    def test_sparse_data(self):
        sparse_data = self.data.to_sparse()
        self.send_signal(self.widget.Inputs.data, sparse_data)
        self.__init_widget()

        self.widget.selection = [((-600000000, -590000000), "Foo"),
                                 ((-500000000, -490000000), "Bar")]
        self.widget.commit.now()

        data = self.get_output(self.widget.Outputs.selected_data)
        self.assertIn("Slice", data.domain)
        self.assertTupleEqual(data.domain["Slice"].values, ("Foo", "Bar"))
        self.assertEqual(len(data), 8)
        self.assertTrue(all(data.get_column(self.widget.x_var) <= -490000000))
        self.assertTrue(all(data.get_column(self.widget.x_var) >= -600000000))
        self.assertFalse(self.widget.Error.non_unique_series.is_shown())

    def test_send_report(self):
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.send_report()
        self.send_signal(self.widget.Inputs.data, None)
        self.widget.send_report()

    def test_visual_settings(self):
        graph = self.widget.graph

        def test_settings():
            font = QFont("Helvetica", italic=True, pointSize=20)
            self.assertFontEqual(
                graph.plot_setter.title_item.item.font(), font
            )

            font.setPointSize(16)
            for item in graph.plot_setter.axis_items:
                self.assertFontEqual(item.label.font(), font)

            font.setPointSize(15)
            for item in graph.plot_setter.axis_items:
                self.assertFontEqual(item.style["tickFont"], font)

            self.assertEqual(
                graph.plot_setter.title_item.item.toPlainText(), "Foo"
            )
            self.assertEqual(graph.plot_setter.title_item.text, "Foo")

            for line in graph.mean_items:
                pen: QPen = line.opts["pen"]
                self.assertEqual(pen.width(), 10)

            for line in graph.line_items:
                pen: QPen = line.opts["pen"]
                self.assertEqual(pen.width(), 3)
                self.assertEqual(pen.color().alpha(), 220)

        self.send_signal(self.widget.Inputs.data, self.data)
        key, value = ("Fonts", "Font family", "Font family"), "Helvetica"
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Title", "Font size"), 20
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Title", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis title", "Font size"), 16
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis title", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Axis ticks", "Font size"), 15
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Axis ticks", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Fonts", "Legend", "Font size"), 17
        self.widget.set_visual_settings(key, value)
        key, value = ("Fonts", "Legend", "Italic"), True
        self.widget.set_visual_settings(key, value)

        key, value = ("Annotations", "Title", "Title"), "Foo"
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Mean", "Width"), 10
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Lines", "Width"), 3
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Lines", "Opacity"), 220
        self.widget.set_visual_settings(key, value)

        key, value = ("Figure", "Range", "Opacity"), 100
        self.widget.set_visual_settings(key, value)

        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()
        self.wait_until_finished()
        test_settings()

        self.send_signal(self.widget.Inputs.data, None)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()
        self.wait_until_finished()
        test_settings()

    def test_slice_picker_enabled(self):
        button = self.widget._slice_picker._SlicePicker__add_button
        self.assertFalse(button.isEnabled())
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertTrue(button.isEnabled())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(button.isEnabled())

    def test_select(self):
        event = Mock()
        event.button.return_value = Qt.LeftButton
        event.buttonDownPos.return_value = Point(60, 480)
        event.pos.return_value = Point(160, 480)
        event.isFinish.return_value = True

        # drag a line before data is sent
        self.widget.graph._view_box.mouseDragEvent(event)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

        # drag a line after data is sent
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()
        self.wait_until_finished()
        self.widget.graph._view_box.mouseDragEvent(event)
        sel_rects = self.widget.graph._SlicerPlot__selection_rect_items
        self.assertEqual(len(sel_rects), 1)
        self.assertEqual(len(self.widget._slice_picker._remove_buttons), 1)
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertIsNotNone(output)

        # click on the plot resets selection
        self.widget.graph._view_box.mouseClickEvent(event)
        sel_rects = self.widget.graph._SlicerPlot__selection_rect_items
        self.assertEqual(len(sel_rects), 0)
        self.assertEqual(len(self.widget._slice_picker._remove_buttons), 0)
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assert_table_equal(output, self.data)

    def test_saved_selection(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()
        self.widget.selection = [((-600000000, -590000000), "Slice A"),
                                 ((-500000000, -490000000), "Slice B")]
        self.widget.commit.now()
        output1 = self.get_output(self.widget.Outputs.selected_data)
        settings = self.widget.settingsHandler.pack_data(self.widget)

        w = self.create_widget(OWSlicer, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.data, widget=w)
        output2 = self.get_output(w.Outputs.selected_data, widget=w)
        self.assert_table_equal(output1, output2)

    def test_retain_selection_on_data_reload(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()
        self.widget.selection = [((-600000000, -590000000), "Slice A"),
                                 ((-500000000, -490000000), "Slice B")]

        self.widget.commit.now()
        output1 = self.get_output(self.widget.Outputs.selected_data)

        self.send_signal(self.widget.Inputs.data, self.data)
        output2 = self.get_output(self.widget.Outputs.selected_data)
        self.assert_table_equal(output1, output2)

    def test_retain_selection_on_smaller_data(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()
        self.widget.selection = [((-600000000, -590000000), "Slice A"),
                                 ((-500000000, -490000000), "Slice B")]
        self.assertEqual(self.widget.selection,
                         [((-600000000, -590000000), "Slice A"),
                          ((-500000000, -490000000), "Slice B")])

        flt = FilterContinuous("Month", FilterContinuous.LessEqual, -495000000)
        filtered_data = Values([flt])(self.data)

        self.send_signal(self.widget.Inputs.data, filtered_data)
        output2 = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(output2), 6)
        self.assertEqual(self.widget.selection,
                         [((-600000000, -590000000), "Slice A"),
                          ((-500000000, -497145600), "Slice B")])

    def assertFontEqual(self, font1: QFont, font2: QFont):
        self.assertEqual(font1.family(), font2.family())
        self.assertEqual(font1.pointSize(), font2.pointSize())
        self.assertEqual(font1.italic(), font2.italic())

    def __init_widget(self, widget=None):
        if not widget:
            widget = self.widget

        model = widget._key_vars_view.model()
        sel_model = widget._key_vars_view.selectionModel()

        selection = QItemSelection()
        selection.select(model.index(0, 0), model.index(1, 0))
        sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

    def test_transformation_added(self):
        # table doesn't have transformations - attribute no transformation is added
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertNotIn(TRANSFORMATIONS_ATTRIBUTE, output.attributes)

        # table has transformations - add domain and aggregation preprocessing
        ti = InfoTransform(self.data.domain)
        self.data.attributes[TRANSFORMATIONS_ATTRIBUTE] = (ti,)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.selected_data)
        transformations = output.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(transformations))

        # info transform also in the beginning of workflow
        self.assertIsInstance(transformations[0], InfoTransform)
        self.assertEqual(self.data.domain, transformations[0].domain)

        # test aggregation transformation
        self.assertIsInstance(transformations[1], SlicerPreprocessor)
        self.assertEqual(self.data.domain["Month"], transformations[1].x_var)
        self.assertIsNone(transformations[1].selection)

        # change x_var and set selection - see if preprocessor cages
        simulate.combobox_activate_index(self.widget.controls.x_var, 1)
        self.widget.selection = [((-600000000, -590000000), "Foo"),
                                 ((-500000000, -490000000), "Bar")]
        self.widget.commit.deferred()
        output = self.get_output(self.widget.Outputs.selected_data)
        transformations = output.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(transformations))

        self.assertIsInstance(transformations[0], InfoTransform)
        self.assertEqual(self.data.domain, transformations[0].domain)
        self.assertIsInstance(transformations[1], SlicerPreprocessor)
        self.assertEqual(self.data.domain["Air passengers"],
                         transformations[1].x_var)
        self.assertListEqual(
            [((-600000000, -590000000), "Foo"),
             ((-500000000, -490000000), "Bar")], transformations[1].selection
        )

    def test_duplicate_values(self):
        self.send_signal(self.widget.Inputs.data, self.data)
        self.__init_widget()

        self.widget.selection = [((-600000000, -590000000), "Slice A"),
                                 ((-500000000, -490000000), "Slice A")]
        self.widget.commit.deferred()
        self.assertTrue(self.widget.Error.domain_duplicates.is_shown())

        self.widget.selection = [((-600000000, -590000000), "Slice A"),
                                 ((-500000000, -490000000), "Slice B")]
        self.widget.commit.deferred()
        self.assertFalse(self.widget.Error.domain_duplicates.is_shown())


class TestSlicerPreprocessor(unittest.TestCase):
    def setUp(self):
        domain = Domain([ContinuousVariable("a"), ContinuousVariable("b")])
        x = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 3], [2, 5]]
        self.data = Table.from_list(domain, x)

    def test_slicing(self):
        slices = [((0.1, 2.2), "Slice 1"), ((2.3, 5), "Slice 2")]
        sp = SlicerPreprocessor(self.data.domain["a"], slices)
        result = sp(self.data)

        self.assertTupleEqual(self.data.domain.attributes, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("Slice 1", "Slice 2"), result.domain.metas[0].values)

        expected = np.array([[1, 2], [2, 3], [1, 3], [2, 5], [3, 4], [4, 5]])
        np.testing.assert_array_equal(expected, result.X)
        expected = np.array([[0], [0], [0], [0], [1], [1]])
        np.testing.assert_array_equal(expected, result.metas)

        slices = [((0.1, 1.1), "Slice 1"), ((2.3, 5), "Slice 2")]
        sp = SlicerPreprocessor(self.data.domain["a"], slices)
        result = sp(self.data)

        self.assertTupleEqual(self.data.domain.attributes, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("Slice 1", "Slice 2"), result.domain.metas[0].values)

        expected = np.array([[1, 2], [1, 3], [3, 4], [4, 5]])
        np.testing.assert_array_equal(expected, result.X)
        np.testing.assert_array_equal(np.array([[0], [0], [1], [1]]), result.metas)

        slices = [((6, 7), "Slice 1"), ((8, 9), "Slice 2")]
        sp = SlicerPreprocessor(self.data.domain["a"], slices)
        result = sp(self.data)

        self.assertTupleEqual(self.data.domain.attributes, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("Slice 1", "Slice 2"), result.domain.metas[0].values)

        self.assertTupleEqual((0, 2), result.X.shape)
        self.assertTupleEqual((0, 1), result.metas.shape)

    def test_missing_attribute(self):
        slices = [((0.1, 2.2), "Slice 1"), ((2.3, 5), "Slice 2")]
        sp = SlicerPreprocessor(ContinuousVariable("c"), slices)
        with self.assertRaises(ValueError) as err:
            sp(self.data)
        self.assertEqual(
            "The Series Slicer transformation expects data to contain c variable, "
            "which is missing in the data.",
            str(err.exception),
        )

    def test_repr(self):
        slices = [((0.1, 2.2), "Slice 1"), ((2.3, 5), "Slice 2")]
        sp = SlicerPreprocessor(self.data.domain["a"], slices)
        expected = (
            f"<h4>Series Slicer</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Axis x: </th><td>a</td></tr><tr><th>Slices: </th>"
            "<td>Slice 1: 0.10 - 2.20<br>Slice 2: 2.30 - 5.00</td></tr></table></div>"
        )
        self.assertEqual(expected, str(sp))

        slices = [((6, 7), "Slice 1"), ((8, 9), "Slice 2")]
        sp = SlicerPreprocessor(self.data.domain["b"], slices)
        expected = (
            f"<h4>Series Slicer</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Axis x: </th><td>b</td></tr><tr><th>Slices: </th>"
            "<td>Slice 1: 6.00 - 7.00<br>Slice 2: 8.00 - 9.00</td></tr></table></div>"
        )
        self.assertEqual(expected, str(sp))

        sp = SlicerPreprocessor(self.data.domain["b"], None)
        expected = (
            f"<h4>Series Slicer</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Axis x: </th><td>b</td></tr>"
            "<tr><th>Slices: </th><td>N/A</td></tr></table></div>"
        )
        self.assertEqual(expected, str(sp))


if __name__ == "__main__":
    unittest.main()
