import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt

from Orange.data import FileFormat, dataset_dirs, Table
from Orange.data.io import TabReader
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.utils.filedialogs import format_filter
from numpy.testing import assert_array_equal, assert_array_almost_equal

from orangecontrib.experiment_analytics.widgets.owmultifile import OWMultifile, PANDAS_READER_FORMATS


class TestOWMultifile(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWMultifile)  # type: OWMultifile

    def load_files(self, *files, reader=None):
        files = [FileFormat.locate(name, dataset_dirs) for name in files]

        def open_with_no_specific_format(a, b, c, filters, e):
            return files, filters.split(";;")[0]

        def open_with_specific_format(a, b, c, filters, e):
            return files, format_filter(reader)

        patchfn = open_with_no_specific_format if reader is None \
            else open_with_specific_format

        # pretend that files were chosen in the open dialog
        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileNames", patchfn):
            self.widget.browse_files()

    def test_load_files(self):
        self.load_files("iris", "titanic")
        out = self.get_output("Data")
        iris = Table("iris")
        titanic = Table("titanic")
        for a in list(iris.domain.variables) + list(titanic.domain.variables):
            self.assertIn(a, out.domain)
        self.assertEqual(
            set(out.domain.class_vars),
            set(iris.domain.class_vars) | set(titanic.domain.class_vars))
        self.assertEqual(len(out), len(iris) + len(titanic))

    def test_load_files_reader(self):
        self.load_files("iris")
        self.assertIs(self.widget.recent_paths[0].file_format, None)
        self.load_files("zoo", reader=TabReader)
        self.assertEqual(self.widget.recent_paths[1].file_format,
                         "Orange.data.io.TabReader")

    def test_filename(self):
        self.load_files("iris", "titanic")
        out = self.get_output("Data")
        iris = out[:len(Table("iris"))]
        fns = set([e["Filename"].value for e in iris])
        self.assertTrue(len(fns), 1)
        self.assertIn("iris", fns.pop().lower())
        titanic = out[len(Table("iris")):]
        fns = set([e["Filename"].value for e in titanic])
        self.assertTrue(len(fns), 1)
        self.assertIn("titanic", fns.pop().lower())

    def test_load_clear(self):
        self.load_files("iris")
        self.load_files("titanic")
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("iris")) + len(Table("titanic")))
        self.widget.clear()
        out = self.get_output("Data")
        self.assertIsNone(out)
        self.load_files("iris", "titanic")
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertEqual(len(out), len(Table("titanic")))
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        out = self.get_output("Data")
        self.assertIsNone(out)

    def test_saving_setting(self):
        self.load_files("iris")
        self.load_files("zoo", reader=TabReader)
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.widget = self.create_widget(OWMultifile, stored_settings=settings)
        self.assertEqual(self.widget.recent_paths[0].basename, "iris.tab")
        self.assertEqual(self.widget.recent_paths[0].file_format, None)
        self.assertEqual(self.widget.recent_paths[1].basename, "zoo.tab")
        self.assertEqual(self.widget.recent_paths[1].file_format,
                         "Orange.data.io.TabReader")

    def test_files_relocated_on_saved_workflow(self):
        tempdir = tempfile.mkdtemp()
        try:
            oiris = FileFormat.locate("iris.tab", dataset_dirs)
            ciris = os.path.join(tempdir, "iris.tab")
            shutil.copy(oiris, ciris)
            with patch("Orange.widgets.widget.OWWidget.workflowEnv",
                       Mock(return_value={"basedir": tempdir})):
                self.load_files(ciris)
                self.assertEqual(self.widget.recent_paths[0].relpath,
                                 "iris.tab")
        finally:
            shutil.rmtree(tempdir)

    def test_files_relocated_after_workflow_save(self):
        tempdir = tempfile.mkdtemp()
        try:
            oiris = FileFormat.locate("iris.tab", dataset_dirs)
            ciris = os.path.join(tempdir, "iris.tab")
            shutil.copy(oiris, ciris)
            self.load_files(ciris)
            self.assertEqual(self.widget.recent_paths[0].relpath, None)
            with patch("Orange.widgets.widget.OWWidget.workflowEnv",
                       Mock(return_value={"basedir": tempdir})):
                self.widget.workflowEnvChanged("basedir", tempdir, None)
                self.assertEqual(self.widget.recent_paths[0].relpath,
                                 "iris.tab")
        finally:
            shutil.rmtree(tempdir)

    def test_saving_domain_edit(self):
        self.load_files("iris")
        model = self.widget.domain_editor.model()
        model.setData(model.createIndex(0, 2), "skip", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(data.domain.attributes))
        # saving settings
        settings = self.widget.settingsHandler.pack_data(self.widget)
        # reloading
        self.widget = self.create_widget(OWMultifile, stored_settings=settings)
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(data.domain.attributes))

    def test_reset_domain_edit(self):
        self.load_files("iris")
        model = self.widget.domain_editor.model()
        model.setData(model.createIndex(0, 2), "skip", Qt.EditRole)
        self.widget.apply_button.click()
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(3, len(data.domain.attributes))
        self.widget.reset_domain_edit()
        data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(4, len(data.domain.attributes))

    def test_report_on_empty(self):
        self.widget.send_report()

    def test_report_files(self):
        self.load_files("iris", "zoo")
        self.widget.send_report()

    def test_missing_files_do_not_disappear(self):
        tempdir = tempfile.mkdtemp()
        try:
            oiris = FileFormat.locate("iris.tab", dataset_dirs)
            ciris = os.path.join(tempdir, "iris.tab")
            shutil.copy(oiris, ciris)
            self.load_files(ciris)
            settings = self.widget.settingsHandler.pack_data(self.widget)
        finally:
            shutil.rmtree(tempdir)
        self.widget = self.create_widget(OWMultifile, stored_settings=settings)
        assert not os.path.exists(ciris)
        self.assertEqual(1, len(self.widget.recent_paths))
        self.assertTrue(self.widget.Error.file_not_found.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertEqual("File not found.", self.widget.lb.item(0).toolTip())
        self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

    def test_pandas_missing_file(self):
        def patchfn(a, b, c, filters, e):
            return ["aaa.csv"], filters.split(";;")[0]

        with patch("AnyQt.QtWidgets.QFileDialog.getOpenFileNames", patchfn):
            self.widget.browse_files()
        self.assertTrue(self.widget.Error.file_not_found.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertEqual("File not found.", self.widget.lb.item(0).toolTip())
        self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

    def test_reader_not_found_error(self):
        self.load_files("iris")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        func = "orangecontrib.experiment_analytics.widgets.owmultifile._get_reader"
        with patch(func, side_effect=Exception()):
            self.widget.load_data()
            self.assertTrue(self.widget.Error.missing_reader.is_shown())
            self.assertIsNone(self.get_output(self.widget.Outputs.data))
            self.assertEqual("Reader not found.",
                             self.widget.lb.item(0).toolTip())
            self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

    def test_unknown_reader_error(self):
        self.load_files("iris")
        self.assertIsNotNone(self.get_output(self.widget.Outputs.data))
        with patch("Orange.data.io.TabReader.read",
                   side_effect=Exception("test")):
            self.widget.load_data()
            self.assertTrue(self.widget.Error.read_error.is_shown())
            self.assertIsNone(self.get_output(self.widget.Outputs.data))
            self.assertEqual("Read error:\ntest",
                             self.widget.lb.item(0).toolTip())
            self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

        # test for pandas loader
        self.widget.clear()
        dir_name = os.path.join(os.path.dirname(__file__), "..", "..",
                                "..", "..", "datasets")
        self.load_files(
            os.path.join(dir_name, "airpassengers.csv"),
            os.path.join(dir_name, "airpassengers1.csv"),
        )
        # since copy of pd.read_csv in list, it is not possible to patch it
        PANDAS_READER_FORMATS[1] = (
            PANDAS_READER_FORMATS[1][0],
            Mock(side_effect=Exception("test")),
        )
        self.widget.load_data()
        self.assertTrue(self.widget.Error.read_error.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.data))
        self.assertEqual("Read error:\ntest", self.widget.lb.item(0).toolTip())
        self.assertEqual(Qt.red, self.widget.lb.item(0).foreground())

    def test_load_pandas_load(self):
        def write_dataframe(df_, path_):
            if path_.endswith((".xlsx",)):
                df_.to_excel(path_, index=False)
            else:
                sep = "\t" if ".tsv" in path_ else ","
                df_.to_csv(path_, sep=sep, index=False)

        dir_name = os.path.join(os.path.dirname(__file__), "..", "..",
                                "..", "..", "datasets")
        df = pd.read_csv(os.path.join(dir_name, "airpassengers.csv"))
        df1 = pd.read_csv(os.path.join(dir_name, "airpassengers1.csv"))

        with tempfile.TemporaryDirectory() as tmpdir:
            x = y = metas = None
            for mf in ("csv", "tsv", "xlsx"):
                compressions = ("",)
                if mf not in ("xlsx",):
                    compressions += ("_bz2", "_gz", "_xz")
                for comp in compressions:
                    extension = f"{mf}{comp.replace('_', '.')}"

                    path = os.path.join(tmpdir, f"airpassengers.{extension}")
                    path1 = os.path.join(tmpdir, f"airpassengers1.{extension}")
                    write_dataframe(df, path)
                    write_dataframe(df1, path1)

                    self.load_files(path, path1)
                    out = self.get_output("Data")
                    self.assertEqual(len(out.domain.attributes), 2)
                    self.assertEqual(out.domain.attributes[0].have_date, 1)
                    self.assertEqual(len(out), 156)
                    self.assertFalse(np.isnan(out.X).any())
                    self.assertTupleEqual(
                        out.domain["Filename"].values,
                        (f"airpassengers.{extension}", f"airpassengers1.{extension}"),
                    )
                    assert_array_equal(
                        out[:, out.domain["Filename"]].metas,
                        np.array([0] * 144 + [1] * 12).reshape(-1, 1),
                    )
                    self.widget.clear()
                    if x is not None:
                        assert_array_almost_equal(out.X, x, decimal=10)
                        assert_array_almost_equal(out.Y, y, decimal=10)
                        assert_array_almost_equal(out.metas, metas, decimal=10)
                    else:
                        x, y, metas = out.X, out.Y, out.metas

    def test_filename_path(self):
        dir_name = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        dir_name = os.path.join(os.path.normpath(dir_name), "datasets")
        name1, name2 = "airpassengers.csv", "airpassengers1.csv"
        self.load_files(os.path.join(dir_name, name1),
                        os.path.join(dir_name, name2))

        out = self.get_output("Data")
        filename_var = out.domain["Filename"]
        self.assertEqual(filename_var.values, (name1, name2))
        self.assertEqual(filename_var.attributes, {"origin": dir_name.replace("\\", "/")})

    def test_duplicated_files(self):
        self.load_files("iris", "iris")
        self.assertTrue(self.widget.Error.duplicated_files.is_shown())
        self.widget.lb.item(0).setSelected(True)
        self.widget.remove_item()
        self.assertFalse(self.widget.Error.duplicated_files.is_shown())


if __name__ == "__main__":
    unittest.main()
