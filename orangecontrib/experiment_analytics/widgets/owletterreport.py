from contextlib import contextmanager
from itertools import chain
from types import SimpleNamespace
from typing import Optional, List, Tuple, Any, Callable, Dict, Union

import numpy as np
import pandas as pd
from AnyQt.QtCore import QItemSelection, Qt, QSize, QModelIndex, \
    QItemSelectionRange, QItemSelectionModel, Signal, QPoint, QMimeData, \
    QSortFilterProxyModel
from AnyQt.QtGui import QColor, QFont, QPainter, QResizeEvent, QDrag
from AnyQt.QtWidgets import QSizePolicy, QTableView, QHeaderView, \
    QStyleOptionViewItem, QStyledItemDelegate, QMenu, QAction, QLineEdit

from orangewidget.utils.listview import ListViewFilter
from Orange.data import Table, DiscreteVariable, table_to_frame, \
    table_from_frame, ContinuousVariable, Variable, Domain
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, \
    Setting
from Orange.widgets.utils.itemmodels import PyTableModel, DomainModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.widget import OWWidget, Input, Msg

from orangecontrib.experiment_analytics.excel_export import save, BorderRole
from orangecontrib.experiment_analytics.letter_report import simple_letter_report
from orangecontrib.experiment_analytics.widgets.letter_report_widgets import \
    FrozenHeaderTableView, ScrollableColumnTableView

from orangewidget.settings import Context

AggregationRole = next(gui.OrangeUserRole)
LettersRole = next(gui.OrangeUserRole)
MAX_GROUPS = 26
MEAN, TB = range(1, 3)


@contextmanager
def disconnected(signal: Signal, slot: Callable,
                 type_: Qt.ConnectionType = Qt.UniqueConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, type_)


class Results(SimpleNamespace):
    grouped_data_values: Optional[np.ndarray] = None
    header_data: List[List[Union[str, int]]] = []
    header_span_data: Optional[np.ndarray] = None
    data: Dict[str, List[int]] = {}
    role_data: Dict[str, List[Dict]] = {}


class BorderedItemDelegate(QStyledItemDelegate):
    def paint(
            self,
            painter: QPainter,
            option: QStyleOptionViewItem,
            index: QModelIndex
    ):
        QStyledItemDelegate.paint(self, painter, option, index)
        if index.data(BorderRole):
            painter.save()
            painter.setPen(QColor(Qt.darkGray))
            rect = option.rect
            painter.drawLine(rect.topLeft(), rect.topRight())
            painter.restore()


class LetterReportVariablesModel(PyTableModel):
    MIME_TYPE = "application/x-Orange-LetterReportVariablesModel"

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return super().data(index, role)

        if index.column() in (MEAN, TB):
            if role == Qt.DisplayRole:
                return "Mean" if index.column() == MEAN else "% TB"

            elif role == Qt.CheckStateRole:
                checked = super().data(index, Qt.EditRole)
                return Qt.Checked if checked else Qt.Unchecked

            elif role == Qt.ToolTipRole:
                return super().data(index.siblingAtColumn(0), role)

        return super().data(index, role)

    def setData(self, index: QModelIndex, value: Any,
                role: int = Qt.EditRole) -> bool:
        if not index.isValid():
            return super().setData(index, value, role)

        if index.column() in (MEAN, TB):
            if role == Qt.CheckStateRole:
                return super().setData(index, bool(value), role=Qt.EditRole)

        return super().setData(index, value, role)

    def flags(self, index: QModelIndex) -> int:
        flags_ = super().flags(index)
        flags_ |= Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled
        if not index.isValid() or index.column() not in (MEAN, TB):
            return flags_
        return flags_ | Qt.ItemIsUserCheckable

    @staticmethod
    def supportedDropActions() -> int:
        return Qt.MoveAction

    def mimeTypes(self):
        return [self.MIME_TYPE]

    def mimeData(self, indexes: List[QModelIndex]) -> QMimeData:
        if len(indexes) == 0:
            return None

        mime = QMimeData()
        mime.setData(self.MIME_TYPE, b"see properties: _items")
        mime.setProperty("_items", self[indexes[0].row()])
        return mime

    def dropMimeData(self, mime: QMimeData, action: Qt.DropAction,
                     row: int, _, index: QModelIndex) -> bool:
        if action != Qt.MoveAction or not mime.hasFormat(self.MIME_TYPE):
            return False

        row = max(index.row(), row)
        if row < 0:
            row = len(self)

        with disconnected(self.dataChanged,
                          self.parent().on_report_var_changed):
            self.insert(row, mime.property("_items"))
        return True


class LetterReportVariablesView(ScrollableColumnTableView):
    drop_finished = Signal()

    def __init__(self, search: QLineEdit):
        super().__init__(
            selectionMode=QTableView.SingleSelection,
            selectionBehavior=QTableView.SelectRows,
            defaultDropAction=Qt.MoveAction,
            dragDropMode=QTableView.DragDrop,
            dragDropOverwriteMode=False,
            showGrid=False,
        )

        self.__search = search
        self.__search.setPlaceholderText("Filter...")
        self.__search.textChanged.connect(self.__on_search_text_changed)

        self.__proxy_model = QSortFilterProxyModel()
        self.__proxy_model.setFilterKeyColumn(-1)
        self.__proxy_model.setFilterCaseSensitivity(Qt.CaseInsensitive)

    def setModel(self, model: LetterReportVariablesModel):
        self.__proxy_model.setSourceModel(model)
        super().setModel(self.__proxy_model)

    def __on_search_text_changed(self):
        self.__proxy_model.setFilterFixedString(self.__search.text().strip())

    def on_drag_start(self, actions: Qt.DropAction):
        indexes = self.selectionModel().selectedIndexes()
        if len(indexes):
            drag = QDrag(self)
            drag.setMimeData(self.model().mimeData(indexes))
            if drag.exec(actions) == Qt.MoveAction:
                index = self.selectionModel().selectedIndexes()[0]
                self.model().removeRow(index.row())
                self.clearSelection()
                self.drop_finished.emit()


class LetterReportTableModel(PyTableModel):
    def __init__(self, n_decimals: int, **kwargs):
        super().__init__(**kwargs)
        self.__n_decimals = n_decimals

    def set_n_decimals(self, n_decimals: int):
        self.__n_decimals = n_decimals
        self.dataChanged.emit(self.index(0, 1),
                              self.index(self.rowCount() - 1, 1))

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid():
            return super().data(index, role)

        if role == Qt.TextAlignmentRole and index.column() > 0:
            return Qt.AlignCenter

        if role == Qt.DisplayRole:
            agg_role = index.data(role=AggregationRole)
            if agg_role in (MEAN, TB):
                value = super().data(index, role=Qt.EditRole)
                if np.isnan(value):
                    return "?"

                if index.data(AggregationRole) == TB:
                    value *= 100
                text = str(round(value, self.__n_decimals or None))
                if index.data(AggregationRole) == TB:
                    text += "%"

                letters = super().data(index, LettersRole)
                if letters:
                    return text + " (" + letters + ")"
                return text

        return super().data(index, role)


class LetterReportTableView(FrozenHeaderTableView):
    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.stretch_first_column()
        self.resizeRowsToContents()

    def stretch_first_column(self):
        if self.model().columnCount() > 0:
            header: QHeaderView = self.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            width = header.sectionSize(0)
            header.setSectionResizeMode(0, QHeaderView.Interactive)
            self.setColumnWidth(0, width)


def top(arr: np.ndarray) -> float:
    return (arr == np.max(arr)).sum() / len(arr) * 100 if len(arr) else np.nan


def top_box(arr: np.ndarray) -> np.ndarray:
    return (arr == np.max(arr)).astype(int) if len(arr) else arr


def list_(arr: np.ndarray) -> List:
    return list(arr[~np.isnan(arr)])


class ContextHandler(DomainContextHandler):
    @classmethod
    def encode_setting(
            cls,
            context: Context,
            setting: ContextSetting,
            value: Any
    ) -> Tuple:
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                if all(isinstance(e[0], Variable) for e in value):
                    return ([[cls.encode_variable(e[0])] + e[1:]
                             for e in value], -3)
        return super().encode_setting(context, setting, value)

    def decode_setting(
            self,
            setting: ContextSetting,
            value: Any,
            domain: Optional[Domain] = None,
            *args
    ) -> Any:
        def get_var(name):
            if domain is None:
                raise ValueError("Cannot decode variable without domain")
            return domain[name]

        if isinstance(value, tuple):
            data, dtype = value
            if dtype == -3 and data and isinstance(data[0], list):
                return [[get_var(row[0][0])] + row[1:] for row in data]
        return super().decode_setting(setting, value, domain, *args)

    def is_valid_item(
            self,
            setting: ContextSetting,
            item: Any,
            attrs: Dict,
            metas: Dict
    ) -> bool:
        if isinstance(item, list):
            item = item[0]
        return super().is_valid_item(setting, item, attrs, metas)


class OWLetterReport(OWWidget):
    name = "Stacked Letter Report"
    description = "Performs Tukey’s test for equality of means."
    icon = "icons/letterreport.svg"
    priority = 210
    keywords = ["tukey", "test", "compare", "means", "excel", "export"]

    class Inputs:
        data = Input("Data", Table)

    class Error(OWWidget.Error):
        no_cont_features = Msg("At least one numeric feature is required.")
        no_disc_features = Msg("At least one categorical feature is required.")
        not_enough_instances = Msg("At least four instances are required.")
        not_enough_groups = Msg("At least two groups with two "
                                "samples are required.")
        too_many_groups = Msg("Too many groups ({}).")

    settingsHandler = ContextHandler()
    group_vars: List[DiscreteVariable] = ContextSetting([])
    report_vars: List[List] = ContextSetting([], schema_only=True)
    n_decimals: int = Setting(0)
    lines: List[int] = Setting([], schema_only=True)
    auto_apply = Setting(True)

    buttons_area_orientation = Qt.Vertical
    DEFAULT_MEAN_TB = False, False

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self.__cached_result = Results()
        self.__pending_lines: List[int] = self.lines

        self._group_vars_view: ListViewFilter = None
        self._report_vars_view: LetterReportVariablesView = None
        self._view: LetterReportTableView = None

        self._group_vars_model = DomainModel(
            (DomainModel.CLASSES, DomainModel.METAS, DomainModel.ATTRIBUTES),
            valid_types=DiscreteVariable
        )
        self._report_vars_model = LetterReportVariablesModel(parent=self)
        self._model = LetterReportTableModel(self.n_decimals)

        self._invalidate_results()
        self.setup_gui()

        self.settingsAboutToBePacked.connect(self._save_state)

    def _invalidate_results(self):
        self.__cached_result.grouped_data_values = None
        self.__cached_result.header_data = []
        self.__cached_result.header_span_data = None
        self.__cached_result.data = {}
        self.__cached_result.role_data = {}

    def setup_gui(self):
        # Control area
        box = gui.vBox(self.controlArea, "Group by")
        box.setMinimumWidth(300)
        self._group_vars_view = ListViewFilter(
            selectionMode=ListViewFilter.ExtendedSelection,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum),
            minimumSize=QSize(30, 100),
        )
        self._group_vars_view.set_source_model(self._group_vars_model)
        self._group_vars_view.selectionModel().selectionChanged.connect(
            self.__on_group_vars_changed
        )
        box.layout().addWidget(self._group_vars_view)

        box = gui.vBox(self.controlArea, "Variables")
        search = QLineEdit()
        box.layout().addWidget(search)

        self._report_vars_view = LetterReportVariablesView(search)
        self._report_vars_view.setModel(self._report_vars_model)
        self._report_vars_model.dataChanged.connect(self.on_report_var_changed)
        self._report_vars_view.drop_finished.connect(self.__on_drop_finished)
        box.layout().addWidget(self._report_vars_view)

        box.layout().setSpacing(1)
        hbox = gui.hBox(box, False)
        gui.toolButton(hbox, self, "Reset", callback=self.__on_reset)
        gui.rubber(hbox)
        gui.toolButton(hbox, self, "Select All",
                       callback=lambda: self.__on_select_all(True))
        hbox.layout().setSpacing(1)
        gui.toolButton(hbox, self, "Deselect All",
                       callback=lambda: self.__on_select_all(False))

        box = gui.vBox(self.controlArea, "Table")
        gui.spin(box, self, "n_decimals", 0, 5, label="Decimal places:",
                 callback=self.__on_n_decimals_changed)

        # Main area
        self._view = LetterReportTableView(
            selectionMode=QTableView.NoSelection,
            editTriggers=QTableView.NoEditTriggers,
            contextMenuPolicy=Qt.CustomContextMenu,
        )
        self._view.customContextMenuRequested.connect(self.__on_menu_requested)
        self._view.setItemDelegate(BorderedItemDelegate())
        self._view.setWordWrap(True)
        self._view.setModel(self._model)
        self._view.verticalHeader().hide()
        self.mainArea.layout().addWidget(self._view)

        # Buttons area
        gui.button(self.buttonsArea, self, "Export to Excel",
                   callback=self.__on_export_clicked)
        gui.auto_apply(self.buttonsArea, self)

    def __on_group_vars_changed(self):
        rows = self._group_vars_view.selectionModel().selectedRows()
        rows = [self._group_vars_view.model().mapToSource(idx) for idx in rows]
        values = self._group_vars_model[:]
        self.group_vars = [values[row.row()] for row in rows]
        self.commit.deferred()

    def on_report_var_changed(self):
        if self.auto_apply:
            self.__compute_table_data()
            self._setup_table()

    def __on_drop_finished(self):
        if self.auto_apply:
            self._setup_table()

    def __on_reset(self):
        domain = self.data.domain if self.data else None
        if domain is not None:
            self.report_vars = self._init_report_vars(domain)
            self._report_vars_model.wrap(self.report_vars)
            self.on_report_var_changed()

    def __on_select_all(self, check):
        self.report_vars = self._report_vars_model.tolist()
        for report_var in self.report_vars:
            report_var[MEAN] = report_var[TB] = check
        self._report_vars_model.wrap(self.report_vars)
        if len(self.report_vars) > 0:
            self.on_report_var_changed()

    def __on_n_decimals_changed(self):
        self._model.set_n_decimals(self.n_decimals)

    def __on_menu_requested(self, point: QPoint):
        index: QModelIndex = self._view.indexAt(point)
        if not index.isValid() or index.row() == self.n_horizontal_header_rows:
            return

        menu = QMenu(self)
        for text, add in (("Add horizontal line", True),
                          ("Remove horizontal line", False)):
            action = QAction(text, self)
            action.triggered.connect(
                lambda *_, show=add: self.__on_toggle_line(index.row(), show)
            )
            menu.addAction(action)
        menu.popup(self._view.viewport().mapToGlobal(point))

    def __on_toggle_line(self, row: int, show: bool):
        for column in range(self._model.columnCount()):
            index = self._model.index(row, column)
            if index.isValid():
                self._model.setData(index, show, BorderRole)

    def __on_export_clicked(self):
        if self._model.rowCount():
            save(self, self._model, self.n_horizontal_header_rows)

    @property
    def n_horizontal_header_rows(self) -> int:
        return len(self.group_vars) + 2

    @property
    def n_vertical_header_rows(self) -> int:
        return 1

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.closeContext()
        self.data = data
        self.clear()
        self.check_data()
        self.init_models()
        self.openContext(self.data)
        self.apply_settings()
        self.commit.now()
        self.apply_lines()

    def clear(self):
        self.group_vars.clear()
        self._group_vars_model.set_domain(None)
        self.report_vars.clear()
        self._model.clear()
        self._invalidate_results()

    def check_data(self):
        self.clear_messages()

        if self.data is not None:
            if self.data.domain.has_continuous_attributes(True, True) == 0:
                self.Error.no_cont_features()
                self.data = None
            elif self.data.domain.has_discrete_attributes(True, True) == 0:
                self.Error.no_disc_features()
                self.data = None
            elif len(self.data) < 4:
                self.Error.not_enough_instances()
                self.data = None

    def init_models(self):
        domain = self.data.domain if self.data else None

        self._group_vars_model.set_domain(domain)
        if self._group_vars_model:
            self.group_vars = self._group_vars_model[:1]

        self._report_vars_model.clear()
        if domain is not None:
            self.report_vars = self._init_report_vars(domain)
            self._report_vars_model.wrap(self.report_vars)

            header: QHeaderView = self._report_vars_view.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.Stretch)
            header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self._report_vars_view.update_geometry()

    def apply_settings(self):
        # group by
        selection = QItemSelection()
        sel_model: QItemSelectionModel = self._group_vars_view.selectionModel()
        with disconnected(sel_model.selectionChanged,
                          self.__on_group_vars_changed):
            valid_group_vars = []
            group_vars_model_values = self._group_vars_model[:]
            for var in self.group_vars:
                if var in group_vars_model_values:
                    index = group_vars_model_values.index(var)
                    model_index = self._group_vars_view.model().index(index, 0)
                    selection.append(QItemSelectionRange(model_index))
                    valid_group_vars.append(var)
            self.group_vars = valid_group_vars
            sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

        # variables
        if self.data is not None:
            report_vars = [var for var, _, _ in self.report_vars]
            for attr in self.data.domain.variables + self.data.domain.metas:
                if attr.is_continuous and attr not in report_vars:
                    self.report_vars.append([attr, False, False])
        self._report_vars_model.wrap(self.report_vars)

    @gui.deferred
    def commit(self):
        self.Error.too_many_groups.clear()
        self.Error.not_enough_groups.clear()
        self._model.clear()
        self._invalidate_results()
        if not self.group_vars or not self.data or not self.report_vars:
            return

        self._compute_table()
        self._setup_table()

    def _compute_table(self):
        mask = np.any(np.isfinite(self.data.X), 1)
        df = table_to_frame(self.data[mask], include_metas=True)
        # it can happen that categorical variable contain values that do not exist
        # in data which will cause that group is formed for non-existing value
        # by observed=True will use only observed variables for group keys
        grouper = df.groupby([var.name for var in self.group_vars], observed=True)

        aggregations = {var.name: [list_] for var, _, _ in self.report_vars}
        aggregations[self.report_vars[0][0].name].insert(0, len)
        grouped_data = grouper.agg(aggregations)
        grouped_data = grouped_data[~pd.isna(grouped_data).any(axis=1)]
        if len(grouped_data) > MAX_GROUPS:
            self.Error.too_many_groups(len(grouped_data))
            return
        elif sum(grouped_data.values[:, 0] > 1) < 2:
            self.Error.not_enough_groups(len(grouped_data))
            return

        self.__compute_table_header(grouped_data)
        self.__compute_table_data()

    def __compute_table_header(self, grouped_data: pd.DataFrame):
        header_data = []

        # group by variables
        groups = table_from_frame(grouped_data.iloc[:, :1])
        for var in groups.domain.attributes[:-1]:
            assert isinstance(var, DiscreteVariable)
            col = groups.get_column(var)
            header_data.append([var.str_val(x) for x in col])

        # letters
        header_data.append([chr(x) for x in range(65, 65 + len(grouped_data))])

        # total responses
        header_data.append([int(x) for x in grouped_data.values[:, 0]])

        self.__cached_result.grouped_data_values = grouped_data.values[:, 1:]
        self.__cached_result.header_data = header_data
        self.__cached_result.header_span_data = groups.X[:, :-1].T

    def __compute_table_data(self):
        grouped_data_values = self.__cached_result.grouped_data_values
        assert grouped_data_values is not None

        for i, (var, include_mean, include_tb) in enumerate(self.report_vars):
            if not include_mean and not include_tb:
                continue

            treatments = [np.array(gr) for gr in grouped_data_values[:, i]]
            if include_mean:
                self.__insert_table_row(treatments, var, MEAN)
            if include_tb:
                self.__insert_table_row(treatments, var, TB)

    def __insert_table_row(self, treatments_all: List[np.ndarray],
                           variable: ContinuousVariable, agg_type: int):
        data = self.__cached_result.data
        role_data = self.__cached_result.role_data

        label = self._create_vertical_header_label(variable, agg_type)
        if label in data:
            return

        treatments = []
        for i, treatment in enumerate(treatments_all):
            if agg_type == TB:
                treatment = top_box(treatment)
                treatments_all[i] = treatment
            if len(treatment) > 1:
                treatments.append(treatment)

        if len(treatments) < 2:
            return

        letters: List = simple_letter_report(treatments)
        for i, treatment in enumerate(treatments_all):
            if not len(treatment) > 1:
                letters.insert(i, "")
        assert len(treatments_all) == len(letters)

        data[label] = [np.mean(x) for x in treatments_all]
        role_data[label] = [{AggregationRole: agg_type,
                             LettersRole: letters[i]}
                            for i in range(len(treatments_all))]

        self.__cached_result.data = data
        self.__cached_result.role_data = role_data

    def _setup_table(self):
        if len(self.__cached_result.header_data) == 0:
            return

        assert len(self.__cached_result.header_span_data) + 2 == \
               self.n_horizontal_header_rows

        # header data
        header_labels = [var.name for var in self.group_vars] + \
                        ["", "Base Total Responses"]
        header_table = [[label] + row for label, row in
                        zip(header_labels, self.__cached_result.header_data)]

        # main data
        variable_labels = list(chain.from_iterable(
            [[self._create_vertical_header_label(var, MEAN)] * include_mean +
             [self._create_vertical_header_label(var, TB)] * include_tb
             for var, include_mean, include_tb in self.report_vars]
        ))
        variable_table = [[label] + self.__cached_result.data[label]
                          for label in variable_labels
                          if self.__cached_result.data.get(label) is not None]

        self._model.wrap(header_table + variable_table)

        # additional roles (letters, %)
        i = 0
        for label in variable_labels:
            role_row = self.__cached_result.role_data.get(label)
            if role_row is not None:
                for j, role_cell in enumerate(role_row):
                    ind = self._model.index(i + self.n_horizontal_header_rows,
                                            j + self.n_vertical_header_rows)
                    for role, value in role_cell.items():
                        self._model.setData(ind, value, role)
                i += 1

        # vertical header background color
        # use gray color that is used for highlighting inactive view elements
        palette = self.palette()
        inactive_highlight_color = palette.color(palette.Inactive, palette.Highlight)
        for i in range(len(variable_labels)):
            index = self._model.index(i + self.n_horizontal_header_rows, 0)
            self._model.setData(index, inactive_highlight_color, Qt.BackgroundRole)

        # horizontal header fonts
        bold_font = QFont()
        bold_font.setBold(True)
        italic_font = QFont()
        italic_font.setItalic(True)
        for i in range(self.__cached_result.header_span_data.shape[1]):
            col = i + self.n_vertical_header_rows
            index = self._model.index(self.n_horizontal_header_rows - 2, col)
            self._model.setData(index, bold_font, Qt.FontRole)
            index = self._model.index(self.n_horizontal_header_rows - 1, col)
            self._model.setData(index, italic_font, Qt.FontRole)
        index = self._model.index(self.n_horizontal_header_rows - 1, 0)
        font = QFont()
        font.setItalic(True)
        font.setBold(True)
        self._model.setData(index, font, Qt.FontRole)

        # view settings
        self._view.horizontalHeader().setMinimumSectionSize(100)
        self._view.stretch_first_column()
        self._view.set_header_spans(self.__cached_result.header_span_data,
                                    self.n_vertical_header_rows)
        self._view.n_header_rows = self.n_horizontal_header_rows
        self._view.resizeRowsToContents()

    def apply_lines(self):
        for row in self.__pending_lines:
            self.__on_toggle_line(row, True)
        self.__pending_lines.clear()

    def _save_state(self):
        self.lines = [row for row in range(self._model.rowCount())
                      if self._model.index(row, 0).data(BorderRole)]

    def send_report(self):
        if self.data:
            self.report_table("Report", self._model)

    @staticmethod
    def _init_report_vars(domain: Domain) -> List[List]:
        return [[attr] + list(OWLetterReport.DEFAULT_MEAN_TB)
                for attr in domain.variables + domain.metas
                if attr.is_continuous]

    @staticmethod
    def _create_vertical_header_label(
            variable: ContinuousVariable,
            agg_type: int
    ) -> str:
        postfix = "Mean" if agg_type == MEAN else "% TB"
        return f"{variable.name} - {postfix}"


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    table_ = Table("heart_disease")
    WidgetPreview(OWLetterReport).run(table_)
