from itertools import chain, count
from typing import Optional, Callable, List, Tuple, Union, Any

import numpy as np
from AnyQt.QtCore import QItemSelection, Qt, QSize, QModelIndex
from AnyQt.QtGui import QColor
from AnyQt.QtWidgets import QSizePolicy, QTableView, QHeaderView, \
    QStyleOptionViewItem
from scipy.stats import f_oneway

from orangewidget.utils.listview import ListViewSearch
from Orange.data import Table, ContinuousVariable, DiscreteVariable
from Orange.widgets import gui
from Orange.widgets.settings import ContextSetting, DomainContextHandler, \
    Setting
from Orange.widgets.utils.itemmodels import VariableListModel, PyTableModel
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.visualize.owboxplot import SortProxyModel
from Orange.widgets.widget import OWWidget, Input, Msg

from orangecontrib.experiment_analytics.letter_report import letter_report


class ItemDelegate(gui.ColoredBarItemDelegate):
    def __init__(self, minimum: float, maximum: float, **kwargs):
        super().__init__(**kwargs)
        self.__min = minimum
        self.__max = maximum

    def get_bar_ratio(
            self,
            option: QStyleOptionViewItem,
            index: QModelIndex
    ) -> Tuple[Union[float, str], bool]:
        ratio = index.data(gui.BarRatioRole)
        is_float = isinstance(ratio, float) and np.isfinite(ratio)
        if is_float:
            if self.__max != self.__min:
                ratio = (ratio - self.__min) / (self.__max - self.__min)
            if not np.isfinite(ratio):
                ratio = self.__max
        return ratio, is_float


class TableModel(PyTableModel):
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if role in (gui.BarRatioRole,):
            string = super().data(index, Qt.DisplayRole)
            return float(string) if string != "nan" else string
        return super().data(index, role)


class OWCompareMeans(OWWidget):
    name = "Compare Means"
    description = "Pairwise comparison of means using Tukey's test."
    icon = "icons/comparemeans.svg"
    priority = 200
    keywords = ["letter", "report", "tukey", "test"]

    class Inputs:
        data = Input("Data", Table)

    class Error(OWWidget.Error):
        no_cont_features = Msg("At least one numeric feature is required.")
        no_disc_features = Msg("At least one categorical feature is required.")
        not_enough_instances = Msg("At least four instances are required.")
        not_enough_treatments = Msg("At least two treatments with two "
                                    "samples are required.")

    settingsHandler = DomainContextHandler()
    value_var: ContinuousVariable = ContextSetting(None)
    order_by_importance: bool = Setting(False)
    group_var: DiscreteVariable = ContextSetting(None)
    order_grouping_by_importance: bool = Setting(False)
    auto_apply = Setting(True)

    def __init__(self):
        super().__init__()
        self.data: Optional[Table] = None
        self.variable_name = ""
        self.result_anova = ""

        self._value_var_model = VariableListModel()
        self._group_var_model = VariableListModel()
        self._model = TableModel()

        self._value_var_view: ListViewSearch = None
        self._group_var_view: ListViewSearch = None
        self._view: QTableView = None

        self.setup_gui()

    def setup_gui(self):
        # Control area
        sorted_model = SortProxyModel(sortRole=Qt.UserRole)
        sorted_model.setSourceModel(self._value_var_model)
        sorted_model.sort(0)

        view = self._value_var_view = ListViewSearch()
        view.setModel(sorted_model)
        view.setMinimumSize(QSize(30, 100))
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        view.selectionModel().selectionChanged.connect(
            self.__on_value_var_changed
        )

        sorted_model = SortProxyModel(sortRole=Qt.UserRole)
        sorted_model.setSourceModel(self._group_var_model)
        sorted_model.sort(0)

        view = self._group_var_view = ListViewSearch()
        view.setModel(sorted_model)
        view.setMinimumSize(QSize(30, 100))
        view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        view.selectionModel().selectionChanged.connect(
            self.__on_group_var_changed
        )

        box = gui.vBox(self.controlArea, "Group by")
        box.layout().addWidget(self._group_var_view)
        gui.checkBox(box, self, "order_grouping_by_importance",
                     "Order by relevance to variable", visible=False,
                     tooltip="Order by ANOVA over the variables",
                     callback=self.apply_group_var_sorting)

        box = gui.vBox(self.controlArea, "Variable")
        box.layout().addWidget(self._value_var_view)
        gui.checkBox(box, self, "order_by_importance",
                     "Order by relevance to treatments", visible=False,
                     tooltip="Order by ANOVA over the groups",
                     callback=self.apply_value_var_sorting)

        # Main area
        self._view = QTableView(
            sortingEnabled=True,
            selectionMode=QTableView.NoSelection,
            editTriggers=QTableView.NoEditTriggers,
        )
        self._view.verticalHeader().hide()
        self._view.setModel(self._model)

        self.mainArea.layout().addWidget(self._view)

        gui.label(self.mainArea, self, "Variable: %(variable_name)s",
                  alignment=Qt.AlignCenter)
        gui.label(self.mainArea, self, "ANOVA: %(result_anova)s",
                  alignment=Qt.AlignCenter)

        # Buttons area
        gui.auto_apply(self.buttonsArea, self)

    def __on_value_var_changed(self, selection: QItemSelection):
        if not selection:
            return
        self.value_var = selection.indexes()[0].data(gui.TableVariable)
        self.variable_name = self.value_var.name
        self.result_anova = self._compute_anova()
        self.apply_group_var_sorting()
        self.commit.deferred()

    def __on_group_var_changed(self, selection: QItemSelection):
        if not selection:
            return
        self.group_var = selection.indexes()[0].data(gui.TableVariable)
        self.result_anova = self._compute_anova()
        self.apply_value_var_sorting()
        self.commit.deferred()

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.closeContext()
        self.clear()
        self.data = data
        self._check_data()
        self.init_list_view()
        self.openContext(self.data)
        self.set_list_view_selection()
        self.apply_value_var_sorting()
        self.apply_group_var_sorting()
        self.commit.now()

    def _check_data(self):
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

    def init_list_view(self):
        if not self.data:
            return

        domain = self.data.domain
        self._value_var_model[:] = [
            var for var in chain(
                domain.class_vars, domain.metas, domain.attributes)
            if var.is_continuous and not var.attributes.get("hidden", False)]
        self._group_var_model[:] = [
            var for var in chain(
                domain.class_vars, domain.metas, domain.attributes)
            if var.is_discrete and not var.attributes.get("hidden", False)]

        if len(self._value_var_model) > 0:
            self.value_var = self._value_var_model[0]
        else:
            self.value_var = None

        if len(self._group_var_model) > 0:
            self.group_var = self._group_var_model[0]
        else:
            self.group_var = None

        self.variable_name = self.value_var.name if self.value_var else ""
        self.result_anova = self._compute_anova()

    def _compute_anova(self) -> str:
        if self.value_var is None or self.group_var is None:
            return ""
        col = self.data.get_column(self.value_var)
        n_groups = len(self.group_var.values)
        group_col = self.data.get_column(self.group_var)
        groups = (col[group_col == i] for i in range(n_groups))
        groups = (col[~np.isnan(col)] for col in groups)
        groups = [group for group in groups if len(group) > 1]
        if len(groups) < 2:
            return ""
        F, p = f_oneway(*groups)
        return "" if np.isnan(F) else f"{F:.3f} (p={p:.3f})"

    def set_list_view_selection(self):
        for view, var, callback in ((self._value_var_view, self.value_var,
                                     self.__on_value_var_changed),
                                    (self._group_var_view, self.group_var,
                                     self.__on_group_var_changed)):
            src_model = view.model().sourceModel()
            if var not in src_model:
                continue
            sel_model = view.selectionModel()
            sel_model.selectionChanged.disconnect(callback)
            row = src_model.indexOf(var)
            index = view.model().index(row, 0)
            sel_model.select(index, sel_model.ClearAndSelect)
            self.ensure_selection_visible(view)
            sel_model.selectionChanged.connect(callback)

    def apply_value_var_sorting(self):
        def compute_score(attr):
            if attr is group_var:
                return 3
            col = self.data.get_column_view(attr)[0].astype(float)
            groups = (col[group_col == i] for i in range(n_groups))
            groups = (col[~np.isnan(col)] for col in groups)
            groups = [group for group in groups if len(group)]
            p_val = f_oneway(*groups).pvalue.min() if len(groups) > 1 else 2
            if np.isnan(p_val):
                return 2
            return p_val

        if self.data is None:
            return
        group_var = self.group_var
        if self.order_by_importance and group_var is not None:
            n_groups = len(group_var.values)
            group_col = self.data.get_column_view(group_var)[0].astype(float)
            self._sort_list(self._value_var_model, self._value_var_view,
                            compute_score)
        else:
            self._sort_list(self._value_var_model, self._value_var_view, None)

    def apply_group_var_sorting(self):
        def compute_stat(group):
            if group is value_var:
                return 3
            if group is None:
                return -1
            col = self.data.get_column_view(group)[0].astype(float)
            groups = (value_col[col == i] for i in range(len(group.values)))
            groups = (col[~np.isnan(col)] for col in groups)
            groups = [group for group in groups if len(group)]
            p_val = f_oneway(*groups).pvalue.min() if len(groups) > 1 else 2
            if np.isnan(p_val):
                return 2
            return p_val

        if self.data is None:
            return
        value_var = self.value_var
        if self.order_grouping_by_importance:
            value_col = self.data.get_column_view(value_var)[0].astype(float)
            self._sort_list(self._group_var_model, self._group_var_view,
                            compute_stat)
        else:
            self._sort_list(self._group_var_model, self._group_var_view, None)

    @staticmethod
    def _sort_list(source_model: VariableListModel, view: ListViewSearch,
                   key: Optional[Callable] = None):
        if key is None:
            cnt = count()

            def key(_):  # pylint: disable=function-redefined
                return next(cnt)

        for i, attr in enumerate(source_model):
            source_model.setData(source_model.index(i), key(attr), Qt.UserRole)
        OWCompareMeans.ensure_selection_visible(view)

    @staticmethod
    def ensure_selection_visible(view: ListViewSearch):
        selection = view.selectedIndexes()
        if len(selection) == 1:
            view.scrollTo(selection[0])

    def clear(self):
        self._value_var_model[:] = []
        self._group_var_model[:] = []
        self._model.clear()
        self.variable_name = ""
        self.result_anova = ""

    @gui.deferred
    def commit(self):
        self.Error.not_enough_treatments.clear()
        if not self.data:
            return

        names, treatments = self._get_treatments()
        if len(treatments) < 2:
            self.Error.not_enough_treatments()
            self._model.clear()
            return

        letters: List[List[str]] = letter_report(treatments)

        self._setup_table(names, letters, treatments)

    def _get_treatments(self) -> Tuple[List[str], List[np.ndarray]]:
        value_var = self.value_var
        group_var = self.group_var

        group_col = self.data.get_column(group_var)
        value_col = self.data.get_column(value_var)

        names, treatments = [], []
        for i, group_label in enumerate(group_var.values):
            group = value_col[group_col == i]
            group = group[~np.isnan(group)]
            if len(group) > 1:
                names.append(group_label)
                treatments.append(group)

        return names, treatments

    def _setup_table(
            self,
            names: List[str],
            letters: List[List[str]],
            treatments: List[np.ndarray]
    ) -> None:
        assert len(letters) > 0

        counts = [len(t) for t in treatments]
        means = [t.mean() for t in treatments]
        sds = [t.std() for t in treatments]
        rsds = [sd / mean for mean, sd in zip(means, sds)]

        labels = [self.group_var.name] + [""] * len(letters[0]) + \
                 ["#", "Mean", "SD", "RSD"]
        table = [[name] + letters_ + [cnt, mean, sd, rsd]
                 for name, letters_, cnt, mean, sd, rsd in
                 zip(names, letters, counts, means, sds, rsds)]

        self._model.clear()
        self._model.setHorizontalHeaderLabels(labels)
        self._model.wrap(table)

        count_index = self._model.columnCount() - 4
        for i in range(1, count_index):
            self._view.setColumnWidth(i, 30)
            for j in range(self._model.rowCount()):
                ind = self._model.index(j, i)
                self._model.setData(ind, Qt.AlignCenter, Qt.TextAlignmentRole)
        mean_index = self._model.columnCount() - 3
        for i in range(1, mean_index):
            self._view.setItemDelegateForColumn(i, None)

        kwargs = {"parent": self, "color": QColor(70, 190, 250)}
        for i, values in enumerate((means, sds, rsds)):
            values = [v for v in values if np.isfinite(v)] or [0]
            delegate = ItemDelegate(np.min(values), np.max(values), **kwargs)
            self._view.setItemDelegateForColumn(mean_index + i, delegate)

        header: QHeaderView = self._view.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignLeft)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(count_index, QHeaderView.ResizeToContents)
        # Sort by Mean
        header.setSortIndicator(mean_index, Qt.DescendingOrder)
        self._model.sort(mean_index, Qt.DescendingOrder)

    def send_report(self):
        if not self.data:
            return
        self.report_items((("Group by", self.group_var),
                           ("Variable", self.value_var)))
        self.report_table("Report", self._view)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWCompareMeans).run(Table("iris"))
