import os
from functools import reduce, partial
from itertools import chain, count
from collections import Counter, namedtuple
from typing import List, Callable, Optional

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QSizePolicy as Policy, QGridLayout, QLabel, \
    QFileDialog, QStyle, QListWidget

from Orange.data import Domain, Table, DiscreteVariable, Variable, table_from_frame
from Orange.data.io import FileFormat, class_from_qualified_name
from Orange.data.util import get_unique_names_duplicates, get_unique_names
from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting, ContextSetting, \
    PerfectDomainContextHandler, SettingProvider
from Orange.widgets.utils.annotated_data import add_columns
from Orange.widgets.utils.domaineditor import DomainEditor
from Orange.widgets.utils.filedialogs import RecentPathsWidgetMixin, \
    RecentPath, open_filename_dialog
from Orange.widgets.utils.signals import Output


PANDAS_READER_FORMATS = [
    (("tsv", "tsv.gz", "tsv.bz2", "tsv.xz"), partial(pd.read_csv, delimiter="\t")),
    (("csv", "csv.gz", "csv.bz2", "csv.xz"), pd.read_csv),
    (("xls", "xlsx"), pd.read_excel),
]
PANDAS_SUPPORTED_EXT = tuple(e for exts, _ in PANDAS_READER_FORMATS for e in exts)


def concatenate_data(tables, filenames, path):
    if not tables:
        return None

    domain = _merge_domains([table.domain for table in tables])
    name = get_unique_names(domain, "Filename")
    source_var = DiscreteVariable(name, values=filenames)
    source_var.attributes["origin"] = path
    domain = add_columns(domain, metas=(source_var,))

    tables = [table.transform(domain) for table in tables]
    data = type(tables[0]).concatenate(tables)

    source_ids = np.array(list(chain.from_iterable(
        [i] * len(table) for i, table in enumerate(tables)))
    ).reshape((-1, 1))
    with data.unlocked():
        data[:, source_var] = source_ids
    return data


def _merge_domains(domains):
    def fix_names(part):
        for i, attr, name in zip(count(), part, name_iter):
            if attr.name != name:
                part[i] = attr.renamed(name)

    parts = [_get_part(domains, set.union, part)
             for part in ("attributes", "class_vars", "metas")]
    all_names = [var.name for var in chain(*parts)]
    name_iter = iter(get_unique_names_duplicates(all_names))
    for part in parts:
        fix_names(part)
    return Domain(*parts)


def _get_part(domains, oper, part):
    # keep the order of variables: first compute union or intersections as
    # sets, then iterate through chained parts
    vars_by_domain = [getattr(domain, part) for domain in domains]
    valid = reduce(oper, map(set, vars_by_domain))
    valid_vars = [var for var in chain(*vars_by_domain) if var in valid]
    return _unique_vars(valid_vars)


def _unique_vars(seq: List[Variable]):
    AttrDesc = namedtuple(
        "AttrDesc",
        ("template", "original", "values", "number_of_decimals")
    )

    attrs = {}
    for el in seq:
        desc = attrs.get(el)
        if desc is None:
            attrs[el] = AttrDesc(el, True,
                                 el.is_discrete and el.values,
                                 el.is_continuous and el.number_of_decimals)
            continue
        if desc.template.is_discrete:
            sattr_values = set(desc.values)
            # don't use sets: keep the order
            missing_values = tuple(
                val for val in el.values if val not in sattr_values
            )
            if missing_values:
                attrs[el] = attrs[el]._replace(
                    original=False,
                    values=desc.values + missing_values)
        elif desc.template.is_continuous:
            if el.number_of_decimals > desc.number_of_decimals:
                attrs[el] = attrs[el]._replace(
                    original=False,
                    number_of_decimals=el.number_of_decimals)

    new_attrs = []
    for desc in attrs.values():
        attr = desc.template
        if desc.original:
            new_attr = attr
        elif desc.template.is_discrete:
            new_attr = attr.copy()
            for val in desc.values[len(attr.values):]:
                new_attr.add_value(val)
        else:
            assert desc.template.is_continuous
            new_attr = attr.copy(number_of_decimals=desc.number_of_decimals)
        new_attrs.append(new_attr)
    return new_attrs


def concatenate_pandas_frames(
    dfs: List[pd.DataFrame], filenames: List[str], path: str
) -> Table:
    """
    Concatenates pandas dataframes, transforms them to table and adds filename
    column.
    """
    data = table_from_frame(pd.concat(dfs).reset_index(drop=True))
    # add filename to the domain
    name = get_unique_names(data.domain, "Filename")
    source_var = DiscreteVariable(name, values=filenames)
    source_var.attributes["origin"] = path
    domain = add_columns(data.domain, metas=(source_var,))
    data = data.transform(domain)
    source_ids = np.array(
        list(chain.from_iterable([i] * len(df) for i, df in enumerate(dfs)))
    ).reshape((-1, 1))
    with data.unlocked(data.metas):
        data[:, source_var] = source_ids
    return data


def get_pandas_reader(filename: str) -> Optional[Callable]:
    """Get Pandas read functon based on file name extensions"""
    for exts, reader in PANDAS_READER_FORMATS:
        if filename.endswith(exts):
            return reader


class RelocatablePathsWidgetMixin(RecentPathsWidgetMixin):
    """
    Do not rearrange the file list as the RecentPathsWidgetMixin does.
    """

    def add_path(self, filename, reader):
        """Add (or move) a file name to the top of recent paths"""
        self._check_init()
        recent = RecentPath.create(filename, self._search_paths())
        if reader is not None:
            recent.file_format = reader.qualified_name()
        self.recent_paths.append(recent)

    def select_file(self, n):
        return NotImplementedError


class OWMultifile(widget.OWWidget, RelocatablePathsWidgetMixin):
    name = "Multi File"
    icon = "icons/multifile.svg"
    description = "Read data from input files " \
                  "and send a data table to the output."
    priority = 20
    keywords = ["file", "multi", "read"]

    class Outputs:
        data = Output("Data", Table, doc="Concatenated input files.")

    want_main_area = False

    file_idx = []

    settingsHandler = PerfectDomainContextHandler(
        match_values=PerfectDomainContextHandler.MATCH_VALUES_ALL
    )

    recent_paths: List[RecentPath]
    variables: list

    sheet = Setting(None, schema_only=True)
    recent_paths = Setting([], schema_only=True)
    variables = ContextSetting([], schema_only=True)

    class Error(widget.OWWidget.Error):
        file_not_found = widget.Msg("File(s) not found.")
        missing_reader = widget.Msg("Missing reader(s).")
        read_error = widget.Msg("Read error(s).")
        duplicated_files = widget.Msg("Some file names are duplicated.")

    domain_editor = SettingProvider(DomainEditor)

    def __init__(self):
        widget.OWWidget.__init__(self)
        RelocatablePathsWidgetMixin.__init__(self)
        self.domain = None
        self.data = None
        self.loaded_file = ""
        self.sheets = []

        self.lb = gui.listBox(self.controlArea, self, "file_idx",
                              selectionMode=QListWidget.MultiSelection)
        self.default_foreground = None

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, margin=0, orientation=layout)

        file_button = gui.button(
            None, self, '  ...', callback=self.browse_files, autoDefault=False)
        file_button.setIcon(self.style().standardIcon(
            QStyle.SP_DirOpenIcon))
        file_button.setSizePolicy(Policy.Maximum, Policy.Fixed)
        layout.addWidget(file_button, 0, 0)

        remove_button = gui.button(
            None, self, 'Remove', callback=self.remove_item)

        clear_button = gui.button(
            None, self, 'Clear', callback=self.clear)

        layout.addWidget(remove_button, 0, 1)
        layout.addWidget(clear_button, 0, 2)

        reload_button = gui.button(
            None, self, "Reload", callback=self.load_data, autoDefault=False)
        reload_button.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload))
        reload_button.setSizePolicy(Policy.Fixed, Policy.Fixed)
        layout.addWidget(reload_button, 0, 7)

        self.sheet_box = gui.hBox(None, addToLayout=False, margin=0)
        self.sheet_index = 0
        self.sheet_combo = gui.comboBox(None, self, "sheet_index",
                                        callback=self.select_sheet)
        self.sheet_combo.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_label = QLabel()
        self.sheet_label.setText('Sheet')
        self.sheet_label.setSizePolicy(Policy.MinimumExpanding, Policy.Fixed)
        self.sheet_box.layout().addWidget(self.sheet_label, Qt.AlignLeft)
        self.sheet_box.layout().addWidget(self.sheet_combo, Qt.AlignVCenter)
        layout.addWidget(self.sheet_box, 2, 1)
        self.sheet_box.hide()

        layout.addWidget(self.sheet_box, 0, 5)

        layout.setColumnStretch(3, 2)

        box = gui.widgetBox(self.controlArea, "Columns (Double click to edit)")
        self.domain_editor = DomainEditor(self)
        self.editor_model = self.domain_editor.model()
        box.layout().addWidget(self.domain_editor)

        for rp in self.recent_paths:
            self.lb.addItem(rp.abspath)

        box = gui.hBox(self.controlArea)
        gui.rubber(box)

        gui.button(box, self, "Reset", callback=self.reset_domain_edit)
        self.apply_button = gui.button(
            box, self, "Apply", callback=self.apply_domain_edit)
        self.apply_button.setEnabled(False)
        self.apply_button.setFixedWidth(170)
        self.editor_model.dataChanged.connect(
            lambda: self.apply_button.setEnabled(True))

        self._update_sheet_combo()
        self.load_data()

    def set_label(self):
        self.load_data()

    def _select_active_sheet(self):
        if self.sheet:
            try:
                sheet_list = [s[0] for s in self.sheets]
                idx = sheet_list.index(self.sheet)
                self.sheet_combo.setCurrentIndex(idx)
            except ValueError:
                # Requested sheet does not exist in this file
                self.sheet = None
        else:
            self.sheet_combo.setCurrentIndex(0)

    @staticmethod
    def __get_sheets(rp: RecentPath) -> List[str]:
        """Get all sheets in file"""
        try:
            reader = _get_reader(rp)
            return reader.sheets
        except:
            return []

    def _update_sheet_combo(self):
        sheets = Counter()
        for rp in self.recent_paths:
            sheets.update(self.__get_sheets(rp))
        sheets = sorted(sheets.items(), key=lambda x: x[0])

        self.sheets = [(s, s + " (" + str(n) + ")") for s, n in sheets]

        if len(sheets) < 2:
            self.sheet_box.hide()
            self.sheet = None
        else:
            self.sheets.insert(0, (None, "(None)"))
            self.sheet_combo.clear()
            self.sheet_combo.addItems([s[1] for s in self.sheets])
            self._select_active_sheet()
            self.sheet_box.show()

    def select_sheet(self):
        self.sheet = self.sheets[self.sheet_combo.currentIndex()][0]
        self.load_data()

    def remove_item(self):
        ri = [i.row() for i in self.lb.selectedIndexes()]
        for i in sorted(ri, reverse=True):
            self.recent_paths.pop(i)
            self.lb.takeItem(i)
        self._update_sheet_combo()
        self.load_data()

    def clear(self):
        self.lb.clear()
        while self.recent_paths:
            self.recent_paths.pop()
        self._update_sheet_combo()
        self.load_data()

    def browse_files(self):
        start_file = self.last_path() or os.path.expanduser("~/")

        readers = [f for f in FileFormat.formats if
                   getattr(f, 'read', None) and getattr(f, "EXTENSIONS", None)]
        filenames, reader, _ = \
            open_filename_dialog(start_file, None, readers,
                                 dialog=QFileDialog.getOpenFileNames)

        self.load_files(filenames, reader)

    def load_files(self, filenames, reader):
        if not filenames:
            return

        for f in filenames:
            self.add_path(f, reader)
            self.lb.addItem(f)

        self._update_sheet_combo()
        self.load_data()

    @staticmethod
    def __show_error(li, msg):
        li.setForeground(Qt.red)
        li.setToolTip(msg)

    def __load_data_orange(self) -> Optional[Table]:
        """
        Load data files with Orange readers. Orange readers are used for
        file types except csv, tsv and Excel.
        """
        data_list = []
        fnok_list = []
        path = ""

        for i, rp in enumerate(self.recent_paths):
            fn = rp.abspath
            li = self.lb.item(i)

            if not os.path.exists(fn):
                self.__show_error(li, "File not found.")
                self.Error.file_not_found()
                continue

            try:
                reader = _get_reader(rp)
                assert reader is not None
            except Exception:  # pylint: disable=broad-except
                self.__show_error(li, "Reader not found.")
                self.Error.missing_reader()
                continue

            try:
                if self.sheet in reader.sheets:
                    reader.select_sheet(self.sheet)
                data_list.append(reader.read())
                path, name = os.path.split(fn)
                fnok_list.append(name)
            except Exception as ex:  # pylint: disable=broad-except
                self.__show_error(li, "Read error:\n" + str(ex))
                self.Error.read_error()

        if len(fnok_list) != len(set(fnok_list)):
            self.Error.duplicated_files()

        if (
            data_list
            and not self.Error.file_not_found.is_shown()
            and not self.Error.missing_reader.is_shown()
            and not self.Error.read_error.is_shown()
            and not self.Error.duplicated_files.is_shown()
        ):
            return concatenate_data(data_list, fnok_list, path)

    def __load_data_pandas(self) -> Optional[Table]:
        """
        Load data files with Pandas readers. Pandas readers are used for CSV,
        TSV and Excel files. Pandas readers are used since when using  Orange
        readers, each file is read to the Orange Table separately before
        concatenation. It can lead to multiple columns for the same column name
        when columns are interpreted as a different type for different files.
        """
        dfs = []
        filenames = []
        path = ""

        for i, rp in enumerate(self.recent_paths):
            fn: str = rp.abspath
            li = self.lb.item(i)
            reader = get_pandas_reader(fn)
            try:
                kwargs = {}
                if fn.endswith(("xls", "xlsx")) and self.sheet:
                    sheets = self.__get_sheets(rp)
                    # if sheet in document retrieve selected otherwise take first
                    kwargs["sheet_name"] = self.sheet if self.sheet in sheets else 0
                dfs.append(reader(fn, **kwargs))
                path, name = os.path.split(fn)
                filenames.append(name)
            except FileNotFoundError:
                self.__show_error(li, "File not found.")
                self.Error.file_not_found()
            except Exception as ex:
                self.__show_error(li, "Read error:\n" + str(ex))
                self.Error.read_error()

        if (
            dfs
            and not self.Error.file_not_found.is_shown()
            and not self.Error.read_error.is_shown()
        ):
            return concatenate_pandas_frames(dfs, filenames, path)

    def __all_files_pandas_format(self) -> bool:
        """Check if all files in the list can be read with Pandas."""
        return all(p.abspath.endswith(PANDAS_SUPPORTED_EXT) for p in self.recent_paths)

    def __clear_errors(self):
        self.Error.file_not_found.clear()
        self.Error.missing_reader.clear()
        self.Error.read_error.clear()
        self.Error.duplicated_files.clear()

        for i in range(len(self.recent_paths)):
            li = self.lb.item(i)
            li.setToolTip("")
            if self.default_foreground is None:
                self.default_foreground = li.foreground()
            li.setForeground(self.default_foreground)

    def load_data(self):
        self.closeContext()
        self.__clear_errors()
        if self.__all_files_pandas_format():
            # use pandas reader only when all files can be read with Pandas
            data = self.__load_data_pandas()
        else:
            data = self.__load_data_orange()

        if data is None:
            self.data = None
            self.domain_editor.set_domain(None)
        else:
            self.data = data
            self.openContext(data.domain)
        self.apply_domain_edit()  # sends data

    def storeSpecificSettings(self):
        self.current_context.modified_variables = self.variables[:]

    def retrieveSpecificSettings(self):
        if hasattr(self.current_context, "modified_variables"):
            self.variables[:] = self.current_context.modified_variables

    def apply_domain_edit(self):
        if self.data is None:
            table = None
        else:
            domain, cols = self.domain_editor.get_domain(self.data.domain,
                                                         self.data)
            if not (domain.variables or domain.metas):
                table = None
            else:
                X, y, m = cols
                table = Table.from_numpy(domain, X, y, m, self.data.W)
                table.name = self.data.name
                table.ids = np.array(self.data.ids)
                table.attributes = getattr(self.data, 'attributes', {})

        self.Outputs.data.send(table)
        self.apply_button.setEnabled(False)

    def reset_domain_edit(self):
        self.domain_editor.reset_domain()
        self.apply_domain_edit()

    def send_report(self):
        def get_format_name(format):
            try:
                return format.DESCRIPTION
            except AttributeError:
                return format.__class__.__name__

        if self.data is None:
            self.report_paragraph("File", "No file.")
            return

        files = []

        for rp in self.recent_paths:
            format = _get_reader(rp)
            files.append([rp.abspath, get_format_name(format)])

        self.report_table("Files", table=files)

        self.report_data("Data", self.data)

    def workflowEnvChanged(self, key, value, oldvalue):
        """
        Function called when environment changes (e.g. while saving the scheme)
        It make sure that all environment connected values are modified
        (e.g. relative file paths are changed)
        """
        self.update_file_list(key, value, oldvalue)

    def update_file_list(self, key, value, oldvalue):
        if key == "basedir":
            self._relocate_recent_files()


def _get_reader(rp):
    if rp.file_format:
        reader_class = class_from_qualified_name(rp.file_format)
        return reader_class(rp.abspath)
    else:
        return FileFormat.get_reader(rp.abspath)


if __name__ == "__main__":  # pragma: no cover
    # pylint: disable=ungrouped-imports
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWMultifile).run()
