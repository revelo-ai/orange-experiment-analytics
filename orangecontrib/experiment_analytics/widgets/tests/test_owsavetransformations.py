import contextlib
import os
import tempfile
import unittest
from unittest.mock import patch, mock_open

from AnyQt.QtWidgets import QLabel, QSpacerItem
from numpy.testing import assert_array_equal
from Orange.data import Table
from Orange.preprocess import Normalize
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    ComputeValueTransform,
    InfoTransform,
    HTML_TABLE_STYLE,
)
from orangecontrib.experiment_analytics.widgets.owaggregate import AggregatePreprocessor
from orangecontrib.experiment_analytics.widgets.owsavetransformations import (
    NO_DATA_MESSAGE,
    OWSaveTransformations,
)
from orangecontrib.experiment_analytics.widgets.owslicer import SlicerPreprocessor


@contextlib.contextmanager
def random_temp_file():
    """
    Creates a temporary directory, adds the filename to the path, and deletes
    the directory upon exit.

    Yields
    ------
    The full path to the temporary file.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_filepath = os.path.join(temp_dir, "temp.pkltr")
        yield temp_filepath


class TestOWSaveTransformations(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSaveTransformations)
        self.data_not_valid = Table("iris")
        self.data_valid = self.data_not_valid.copy()
        cvt = InfoTransform(self.data_valid.domain)
        cvt.set_row_count(self.data_valid, self.data_valid)
        self.data_valid.attributes[TRANSFORMATIONS_ATTRIBUTE] = (cvt,)

    @patch("orangecontrib.experiment_analytics.widgets.owsavetransformations.pickle.dump")
    @patch("builtins.open", new_callable=mock_open())
    def test_dataset(self, mock_open_, mock_pickle):
        self.widget.auto_save = True
        self.assertIsNone(self.widget.data)

        self.send_signal(self.widget.Inputs.data, self.data_valid)
        self.assertEqual(self.data_valid.domain, self.widget.data.domain)
        assert_array_equal(self.data_valid.X, self.widget.data.X)
        assert_array_equal(self.data_valid.Y, self.widget.data.Y)
        mock_pickle.reset_mock()
        mock_open_.reset_mock()

        with random_temp_file() as path:
            self.widget.filename = path
            self.widget.auto_save = False
            self.send_signal(self.widget.Inputs.data, self.data_valid)
            self.assertEqual(self.data_valid.domain, self.widget.data.domain)
            assert_array_equal(self.data_valid.X, self.widget.data.X)
            assert_array_equal(self.data_valid.Y, self.widget.data.Y)
            mock_pickle.assert_not_called()
            mock_open_.assert_not_called()

            self.widget.auto_save = True
            self.send_signal(self.widget.Inputs.data, self.data_valid)
            self.assertEqual(self.data_valid.domain, self.widget.data.domain)
            assert_array_equal(self.data_valid.X, self.widget.data.X)
            assert_array_equal(self.data_valid.Y, self.widget.data.Y)
            mock_pickle.assert_called()
            mock_open_.assert_called()

    @patch("orangecontrib.experiment_analytics.widgets.owsavetransformations.pickle.dump")
    @patch("builtins.open", new_callable=mock_open())
    def test_data_not_valid(self, mock_open_, mock_pickle):
        self.widget.auto_save = True
        self.assertIsNone(self.widget.data)

        self.send_signal(self.widget.Inputs.data, self.data_not_valid)
        self.assertEqual(self.data_not_valid, self.widget.data)
        mock_pickle.assert_not_called()
        mock_open_.assert_not_called()

        with random_temp_file() as path:
            self.widget.filename = path
            self.widget.auto_save = False
            self.send_signal(self.widget.Inputs.data, self.data_not_valid)
            self.assertEqual(self.data_not_valid, self.widget.data)
            mock_pickle.assert_not_called()
            mock_open_.assert_not_called()

            self.widget.auto_save = True
            self.send_signal(self.widget.Inputs.data, self.data_not_valid)
            self.assertEqual(self.data_not_valid, self.widget.data)
            mock_pickle.assert_not_called()
            mock_open_.assert_not_called()

    def test_save_file_write(self):
        self.widget.auto_save = True

        with random_temp_file() as path:
            self.widget.filename = path
            self.assertFalse(os.path.isfile(path))
            self.send_signal(self.widget.Inputs.data, self.data_valid)
            self.assertTrue(os.path.isfile(path))

    def test_error(self):
        self.send_signal(self.widget.Inputs.data, self.data_not_valid)
        self.assertTrue(self.widget.Error.no_transformations.is_shown())

        self.send_signal(self.widget.Inputs.data, self.data_valid)
        self.assertFalse(self.widget.Error.no_transformations.is_shown())

        self.send_signal(self.widget.Inputs.data, self.data_not_valid)
        self.assertTrue(self.widget.Error.no_transformations.is_shown())

        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_transformations.is_shown())

    def test_transformations_list(self):
        layout = self.widget.scroll_area.scroll_layout

        # one label widget and one spacer in scroll area
        self.assertEqual(2, layout.count())
        label = layout.itemAt(0).widget().findChild(QLabel).text()
        self.assertEqual("No data on input", label)
        self.assertIsInstance(layout.itemAt(1), QSpacerItem)

        self.send_signal(self.widget.Inputs.data, self.data_not_valid)
        # one label widget and one spacer in scroll area
        self.assertEqual(2, layout.count())
        label = layout.itemAt(0).widget().findChild(QLabel).text()
        self.assertEqual(NO_DATA_MESSAGE, label)
        self.assertIsInstance(layout.itemAt(1), QSpacerItem)

        # data have only starting domain - no transformations
        self.send_signal(self.widget.Inputs.data, self.data_valid)
        self.assertEqual(1, layout.count())
        self.assertIsInstance(layout.itemAt(0), QSpacerItem)

        # data with transformations
        d = self.data_valid.domain
        info = InfoTransform(d)
        info.set_row_count(self.data_valid, self.data_valid)
        ap = AggregatePreprocessor(
            [d["iris"]],
            [],
            [d["petal length"], d["petal width"]],
            [("Sum", "sum")],
            None,
        )
        ap.set_domain(d)
        ap.set_row_count(self.data_valid, self.data_valid)
        sl = SlicerPreprocessor(d["petal length"], [((0, 2), "S1"), ((2.1, 8), "S2")])
        sl.set_domain(d)
        sl.set_row_count(self.data_valid, self.data_valid)
        self.data_valid.attributes[TRANSFORMATIONS_ATTRIBUTE] = (info, sl, ap)
        self.send_signal(self.widget.Inputs.data, self.data_valid)
        self.assertEqual(3, layout.count())
        label = layout.itemAt(0).widget().findChild(QLabel).text()
        expected = (
            f"<h4>Series Slicer</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Axis x: </th><td>petal length</td></tr>"
            "<tr><th>Slices: </th><td>S1: 0.00 - 2.00<br>S2: 2.10 - 8.00</td></tr>"
            "</table></div>"
        )
        self.assertEqual(expected, label)
        label = layout.itemAt(1).widget().findChild(QLabel).text()
        expected = (
            f"<h4>Aggregate</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Rows (Group by): </th><td>iris</td></tr>"
            "<tr><th>Columns (Split by): </th><td>N/A</td></tr>"
            "<tr><th>Values to aggregate: </th><td>petal length, petal width</td></tr>"
            "<tr><th>Aggregations: </th><td>Sum</td></tr>"
            "<tr><th>X variable: </th><td>N/A</td></tr>"
            "</table></div>"
        )
        self.assertEqual(expected, label)
        self.assertIsInstance(layout.itemAt(2), QSpacerItem)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(2, layout.count())
        label = layout.itemAt(0).widget().findChild(QLabel).text()
        self.assertEqual("No data on input", label)
        self.assertIsInstance(layout.itemAt(1), QSpacerItem)

    def test_last_preprocessor_added(self):
        """
        Test if current domain added to the list if transformation happened
        last Experiment Analytics transformation
        """
        # no transformation happened after init - only initial domain in list
        self.send_signal(self.widget.Inputs.data, self.data_valid)
        self.assertEqual(1, len(self.widget.data.attributes[TRANSFORMATIONS_ATTRIBUTE]))

        # only orange transformation after init
        data = Normalize(norm_type=Normalize.NormalizeBySpan)(self.data_valid)
        self.send_signal(self.widget.Inputs.data, data)
        transformations = self.widget.data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(transformations))
        self.assertEqual(data.domain, transformations[-1].domain)

        # no Orange transformation after Experiment Analytics transformation
        d = self.data_valid.domain
        sl = SlicerPreprocessor(d["petal length"], [((0, 2), "S1"), ((2.1, 8), "S2")])
        sl.domain = d
        tr = self.data_valid.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.data_valid.attributes[TRANSFORMATIONS_ATTRIBUTE] = tr + (sl,)
        self.send_signal(self.widget.Inputs.data, self.data_valid)
        transformations = self.widget.data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(transformations))
        self.assertEqual(str(sl), str(transformations[-1]))

        # Orange transformation after Experiment Analytics transformation -
        # ComputeValueTransform added by widget
        data = sl(self.data_valid)
        data = Normalize(norm_type=Normalize.NormalizeBySpan)(data)
        self.send_signal(self.widget.Inputs.data, data)
        transformations = self.widget.data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(3, len(transformations))
        self.assertEqual(str(sl), str(transformations[1]))
        self.assertIsInstance(transformations[-1], ComputeValueTransform)
        self.assertEqual(data.domain, transformations[-1].domain)

    @patch(
        "orangecontrib.experiment_analytics.widgets.owsavetransformations.OWSaveTransformations.report_items"
    )
    @patch(
        "orangecontrib.experiment_analytics.widgets.owsavetransformations.OWSaveTransformations.report_raw"
    )
    def test_report(self, raw_mock, item_mock):
        # report without file path and data
        self.widget.report_button.click()
        raw_mock.assert_not_called()
        item_mock.assert_called_with((("File name", "not set"),))
        item_mock.reset_mock()

        # test with filename set
        self.widget.filename = "foo.pkl"
        self.widget.report_button.click()
        raw_mock.assert_not_called()
        item_mock.assert_called_with((("File name", "foo.pkl"),))
        item_mock.reset_mock()

        # test with not valid data
        self.send_signal(self.widget.Inputs.data, self.data_not_valid)
        self.widget.report_button.click()
        raw_mock.assert_not_called()
        item_mock.assert_called_with((("File name", "foo.pkl"),))
        item_mock.reset_mock()

        # test with data with transformations
        d = self.data_valid.domain
        sl = SlicerPreprocessor(d["petal length"], [((0, 2), "S1"), ((2.1, 8), "S2")])
        data = sl(self.data_valid)
        sl.domain = data.domain

        tr = data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        data.attributes[TRANSFORMATIONS_ATTRIBUTE] = tr + (sl,)
        data = Normalize(norm_type=Normalize.NormalizeBySpan)(data)

        self.send_signal(self.widget.Inputs.data, data)
        self.widget.report_button.click()
        raw_mock.assert_called_with(
            "Transformations",
            f"<h4>Series Slicer</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Axis x: </th><td>petal length</td></tr>"
            "<tr><th>Slices: </th><td>S1: 0.00 - 2.00<br>S2: 2.10 - 8.00</td></tr>"
            "</table></div>"
            f"<hr><h4>Domain transformation</h4>{HTML_TABLE_STYLE}<div><table>"
            f"<tr><th>Changed/added features: </th><td>4</td>"
            f"</tr><tr><th>Unchanged features: </th><td>2</td></tr></table></div>",
        )
        item_mock.assert_called_with((("File name", "foo.pkl"),))

    def test_row_count_warning(self):
        d = self.data_valid.domain
        info = InfoTransform(d)
        info.set_row_count(self.data_valid, self.data_valid)

        # number row changed before save
        sl = SlicerPreprocessor(d["petal length"], [((0, 2), "S1"), ((2.1, 8), "S2")])
        data = sl(self.data_valid)
        sl.set_domain(d)
        sl.set_row_count(self.data_valid, data)
        data = data[:5]
        data.attributes[TRANSFORMATIONS_ATTRIBUTE] = [info, sl]
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.len_changed.is_shown())

        # reset warning - with valid data
        self.send_signal(self.widget.Inputs.data, self.data_valid)
        self.assertFalse(self.widget.Warning.len_changed.is_shown())

        # number row changed before transformation
        data_before = self.data_valid[:5]
        sl = SlicerPreprocessor(
            d["petal length"], [((0, 1.45), "S1"), ((1.45, 8), "S2")]
        )
        data = sl(data)
        sl.set_domain(d)
        sl.set_row_count(data_before, data)
        data.attributes[TRANSFORMATIONS_ATTRIBUTE] = [info, sl]
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.len_changed.is_shown())

        # reset warning - with invalid data
        self.send_signal(self.widget.Inputs.data, self.data_not_valid)
        self.assertFalse(self.widget.Warning.len_changed.is_shown())

        # number row changed before domain transformation
        data_before = self.data_valid[:5]
        cvt = ComputeValueTransform(data_before.domain)
        cvt.set_row_count(data_before, data_before)

        sl = SlicerPreprocessor(
            d["petal length"], [((0, 1.45), "S1"), ((1.45, 8), "S2")]
        )
        data = sl(data_before)
        sl.set_domain(d)
        sl.set_row_count(data_before, data)

        data.attributes[TRANSFORMATIONS_ATTRIBUTE] = [info, cvt, sl]
        self.send_signal(self.widget.Inputs.data, data)
        self.assertTrue(self.widget.Warning.len_changed.is_shown())

        # reset warning - with no data
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Warning.len_changed.is_shown())


if __name__ == "__main__":
    unittest.main()
