import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Set, Tuple, Optional, Union

import numpy as np
from numpy.polynomial.polynomial import polyfit
import pandas as pd
from AnyQt.QtCore import QEvent, QItemSelectionModel, QModelIndex, QSize, Qt
from AnyQt.QtGui import QIcon
from AnyQt.QtWidgets import (
    QCheckBox,
    QLabel,
    QListView,
    QProxyStyle,
    QSizePolicy,
    QStyle,
    QToolButton,
    QLineEdit,
)
from Orange.data.pandas_compat import table_from_frames, table_to_frame
from Orange.data import (
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    Table,
    Variable,
    StringVariable,
    TimeVariable,
)
from Orange.util import wrap_callback
from Orange.widgets import gui
from Orange.widgets.settings import DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.widget import OWWidget
from orangecanvas.gui.utils import disconnected
from orangecontrib.experiment_analytics.aggregate.frequency import frequency
from orangewidget.settings import ContextSetting, Setting
from orangewidget.utils.listview import ListViewSearch
from orangewidget.utils.signals import Input, Output
from orangewidget.widget import Msg

from orangecontrib.experiment_analytics.transformation_export import (
    add_transformation_to_data,
    Transformation,
    create_info_html_table,
)


def auc(df: pd.DataFrame) -> float:
    """
    Area under the curve of points defined with df[:, 0] as x and df[:, 1] as y
    """
    indexes = np.argsort(df.iloc[:, 0].values)
    return np.trapz(df.iloc[indexes, 1], df.iloc[indexes, 0])


def abs_auc(df: pd.DataFrame) -> float:
    """
    Area under the curve of points defined with df[:, 0] as x and absolute
    values of df[:, 1] as y
    """
    indexes = np.argsort(df.iloc[:, 0].values)
    return np.trapz(df.iloc[indexes, 1].abs(), df.iloc[indexes, 0])


def polynomial_function(deg: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit polynomial function on points with x df[:, 0] and y df[:, 1].
    Returns dataframe since the it returns more values per example. DataFrame
    also hold column names.
    """
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    cols = ["Slope", "Intercept"] if deg == 1 else list("abc")
    prefix = "" if deg == 1 else "Quad "
    idx = pd.MultiIndex.from_tuples([(y.name, f"{prefix}{c}") for c in cols])
    return pd.DataFrame(polyfit(x, y, deg=deg)[::-1].reshape(1, -1), columns=idx)


def __frequency_proxy(
    use_damping: bool, compute_r2: bool, df: pd.DataFrame, detrend_degree: int = 2
) -> pd.DataFrame:
    """Change parameters order to fit the frequency function"""
    return frequency(df, use_damping, compute_r2, detrend_degree)


def __frequency_xy(x, y, use_damping=False, compute_r2=False, detrend_degree=1):
    df = pd.DataFrame({"x": x, "y": y})
    return frequency(df, use_damping, compute_r2, detrend_degree)


def custom_function(fun_name: str, fun_text: str, df: pd.DataFrame) -> pd.Series:
    """Evaluates user defined function"""
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    y_name = df.iloc[:, 1].name
    # numpy functions passed as globals - use them without prefix: mean(y)
    res = eval(
        fun_text,
        {f: getattr(np, f) for f in dir(np)},
        {"x": x, "y": y, "frequency": __frequency_xy},
    )
    if isinstance(res, pd.DataFrame):
        # if result is dataframe it already has aggregation column names on
        # level 1 - fix just level 0 names - name of y variable
        cur_y_names = res.columns.get_level_values(0)
        return res.rename({n: y_name for n in cur_y_names}, level=0, axis=1)
    idx = pd.MultiIndex.from_tuples([(y_name, fun_name)])
    return pd.Series([res], index=idx)


X_ATTR_TOOLTIP = "Independent variable for function-based aggregations."

# aggregations that do not depend on x variable selection
AGGREGATIONS_BASIC = {
    "Mean": "mean",
    "Median": "median",
    "Standard deviation": "std",
    "Variance": "var",
    "Sum": "sum",
    "Min": "min",
    "Max": "max",
    "Count defined": "count",
    "Count": "size",
}
# aggregations that need x and y variable
AGGREGATIONS_DEP = {
    "Area under the curve": auc,
    "Absolute area under the curve": abs_auc,
    "Linear fit": partial(polynomial_function, 1),
    "Quadratic fit": partial(polynomial_function, 2),
    "Frequency": __frequency_proxy,
    "Custom function": custom_function,
}
# all aggregations in this dictionary get info button
AGGREGATION_INFO = {
    "Linear fit": (
        "<b>f(x) = mx + b</b><br>"
        "Fit values to be aggregated as a linear<br>"
        "function of the selected <tt>x variable</tt>.<br>"
        "Adds two columns to the output:<br>"
        "<tt>Slope</tt> (<b>m</b>) and <tt>Intercept</tt> (<b>b</b>)."
    ),
    "Quadratic fit": (
        "<b>f(x) = ax<sup>2</sup> + bx + c</b><br>"
        "Fit values to be aggregated as a<br>"
        "quadratic function of the selected<br>"
        "<tt>x variable</tt>.<br>"
        "Adds three columns to the output:<br>"
        "<tt>Quad <b>a</b></tt>, <tt>Quad <b>b</b></tt>, and <tt>Quad <b>c</b></tt>."
    ),
    "Frequency": "Identify the dominant frequency (in Hz) and its amplitude.",
    "freq_damping": (
        "Fit a damped periodic model with exponential decay.<br>"
        "<nobr><tt>y(t) = A * exp(−λt) * cos(ωt+ϕ)</tt></nobr><br>"
        "Add half-life to the aggregated statistics."
    ),
    "freq_r2": (
        "Compute R<sup>2</sup>, which measures how well the single frequency "
        "model fits the data."
    ),
    "Custom function": (
        "You can write any function that accepts vectors <b>x</b> (values of the "
        "selected <tt>x variable</tt>) and <b>y</b> (values to be aggregated).<br>"
        "You can set a custom name for your function, "
        "which is used as the column name in the resulting output table."
    ),
}
# register aggregations which need additional controls
# if tuple it is control - (value, control to be used)
# when string it is just a label
AGG_PARAMS = {
    "Custom function": (
        "f(x, y) =",
        (
            "custom_function",
            partial(gui.lineEdit, placeholderText="Function", stretch=2),
        ),
    ),
    "Frequency": (
        ("freq_damping", partial(gui.checkBox, label="Damping")),
        ("freq_r2", partial(gui.checkBox, label="R2")),
    ),
}


@dataclass
class Result:
    unaggregated_df: pd.DataFrame = None
    result_table: Table = None
    aggregations: Dict[Tuple[Tuple[str, ...], str, str], pd.Series] = field(
        default_factory=dict
    )
    # store warnings for aggregations that cannot be computed
    warnings: Set = field(default_factory=set)


def _table_to_frame(table: Table, result: Result) -> None:
    """
    Transform table to dataframe. Since it is the most time-consuming operation
    (1s for table with shaw (1M, 10)) save resulting dataframe to result object
    so that transformation is performed only once per each data table.
    """
    if result.unaggregated_df is None:
        result.unaggregated_df = table_to_frame(table, include_metas=True)


def _get_cache_key(gb_cols, agg_col, agg, x_attr):
    """
    Generate a key under which aggregations are cached. Key is composed of
    tuple with all groupby columns, name of column which is aggregated,
    aggregation string or tuple, and x attribute if it is one of aggregations
    dependent on it (AGGREGATIONS_DEP)
    """
    agg_type = agg if isinstance(agg, str) else agg[0]  # else tuple
    key_ = (tuple(gb_cols), agg_col, agg)
    if agg_type in AGGREGATIONS_DEP:
        key_ += (x_attr.name,)
    return key_


def _aggregate(
    columns: List[Variable],
    aggregations: List[Tuple[Variable, str, Callable]],
    x_attr: Variable,
    result: Result,
    callback: Callable,
) -> None:
    """
    Perform aggregations and save them in the result object. To speed-up
    aggregations cache aggregations for combination of columns, value_col, and
    aggregation in result obRject.
    """
    result.warnings.clear()
    gb = result.unaggregated_df.groupby(columns, observed=True)

    for i, (col, a, fun) in enumerate(aggregations, start=1):
        key_ = _get_cache_key(columns, col, a, x_attr)
        if key_ not in result.aggregations:
            agg_type = a if isinstance(a, str) else a[0]  # else tuple
            acol = [x_attr.name, col] if agg_type in AGGREGATIONS_DEP else col
            try:
                agg = gb[acol].agg(fun) if isinstance(fun, str) else gb[acol].apply(fun)
            except Exception as ex:
                # there is an overlap between col and x
                if col == x_attr.name:
                    result.warnings.add(
                        "select attribute different for x variable and variables"
                    )
                else:
                    # it will happen when custom function is not working
                    result.warnings.add(str(ex))
                continue

            if isinstance(agg, pd.Series):
                # series objects hold feature name as a name - add aggregation
                agg = agg.rename((col, agg_type))
            else:  # pd.DataFrame
                if agg.index.nlevels > len(columns):
                    # apply add additional column to the index when aggregation
                    # function returns dataframe, it must be dropped for concat
                    agg = agg.reset_index(level=-1, drop=True)
            result.aggregations[key_] = agg
        callback(i / len(aggregations))


def _rename_columns(df: pd.DataFrame) -> None:
    """
    Rename dataframe columns to format: val_name - {unstack col name} - agg name
    """
    # label format:
    df.columns = [
        " - ".join(col[:1] + tuple(f"{c}" for c in col[2:]) + col[1:2])
        for col in df.columns.values
    ]


def _table_from_frame(
    df: pd.DataFrame, input_domain: Domain, row_attrs: List[Variable]
):
    """
    Table to from frame that also takes care of moving variables that were
    in metas before to metas in the resulting table
    """
    df = df.reset_index()

    # pandas in some cases change drops a categorical dtype of row-attributes
    # ensure correct type
    types = {
        StringVariable: "string",
        DiscreteVariable: "category",
        ContinuousVariable: "float",
        TimeVariable: "datetime64[ns]",
    }
    df = df.astype({attr: types[type(input_domain[attr])] for attr in row_attrs})

    # attributes that are in metas in the input table and are row attributes
    # should be in metas in the output table
    orig_metas = {v.name for v in input_domain.metas}
    features = [c for c in df.columns if not (c in row_attrs and c in orig_metas)]
    metas = [c for c in df.columns if c in row_attrs and c in orig_metas]
    return table_from_frames(df[features], df[[]], df[metas])


def _concat_aggregations(
    aggregations: List[Tuple[Variable, str, Callable]],
    row_col_attrs: List[Variable],
    x_attr: Variable,
    result: Result,
):
    """Concatenate aggregations in one dataframe"""
    aggs = []
    for col, a, _ in aggregations:
        key = _get_cache_key(row_col_attrs, col, a, x_attr)
        if key in result.aggregations:
            aggs.append(result.aggregations[key])
    return pd.concat(aggs, axis=1)


def _run(
    data: Table,
    row_attrs: List[Variable],
    col_attrs: List[Variable],
    val_attrs: List[Variable],
    aggregations: List[Tuple[str, Callable]],
    x_attr: Variable,
    result: Result,
    state: Optional[TaskState] = None,
):
    def progress(part):
        if state is not None:
            state.set_progress_value(part * 100)
            if state.is_interruption_requested():
                raise Exception

    row_attrs_ = [x.name for x in row_attrs]
    col_attrs_ = [x.name for x in col_attrs]
    val_attrs_ = [x.name for x in val_attrs]

    if state is not None:
        state.set_status("Aggregating")
    # transform table to dataframe
    _table_to_frame(data, result)
    if state is not None:
        state.set_progress_value(0.2)

    # compute aggregations
    row_col_attrs = row_attrs_ + col_attrs_
    aggregations_ = [(col, a, fun) for col in val_attrs_ for a, fun in aggregations]
    _aggregate(
        row_col_attrs, aggregations_, x_attr, result, wrap_callback(progress, 0.2, 0.8)
    )

    # concatenate results
    new_df = _concat_aggregations(aggregations_, row_col_attrs, x_attr, result)
    progress(0.9)

    # rename columns
    new_df = new_df.unstack(col_attrs_)
    _rename_columns(new_df)

    # transform dataframe to table
    result.result_table = _table_from_frame(new_df, data.domain, row_attrs_)

    # add transformation to dat table
    ap = AggregatePreprocessor(row_attrs, col_attrs, val_attrs, aggregations, x_attr)
    add_transformation_to_data(result.result_table, ap, data)
    progress(1)
    return result


class AggregateListViewSearch(ListViewSearch):
    def selectionCommand(
        self, index: QModelIndex, event: QEvent = None
    ) -> QItemSelectionModel.SelectionFlags:
        flags = super().selectionCommand(index, event)
        selmodel = self.selectionModel()
        if not index.isValid():  # Click on empty viewport; don't clear
            return QItemSelectionModel.NoUpdate
        if selmodel.isSelected(index):
            currsel = selmodel.selectedIndexes()
            if len(currsel) == 1 and index == currsel[0]:
                # Is the last selected index; do not deselect it
                return QItemSelectionModel.NoUpdate
        if (
            event is not None
            and event.type() == QEvent.MouseMove
            and flags & QItemSelectionModel.ToggleCurrent
        ):
            # Disable ctrl drag 'toggle'; can be made to deselect the last
            # index, would need to keep track of the current selection
            # (selectionModel does this but does not expose it)
            flags &= ~QItemSelectionModel.Toggle
            flags |= QItemSelectionModel.Select
        return flags


class InfoButton(QToolButton):
    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.setIcon(QIcon(os.path.join(dir_path, "icons", "info.svg")))
        self.setIconSize(QSize(15, 15))
        # text must be in html to be formatted and to enable line breaks
        self.setToolTip(f"<html><head/><body><p>{text}</p></body></html")
        self.setStyleSheet(
            "QToolButton {padding: 0; border: none;}" "QToolTip {font-size: 14px;}"
        )
        # increase tooltip duration that it does not disappear until mouse is
        # moved from the button
        self.setToolTipDuration(100000)
        # set tooltip delay to 0 to appear immediately
        self.setStyle(self.Ps(self.style()))

    class Ps(QProxyStyle):
        """Proxy style for setting tooltip delay to 0"""

        def styleHint(self, hint, option=None, widget=None, returnData=None):
            if hint == QStyle.SH_ToolTip_WakeUpDelay:
                return 0
            return super().styleHint(hint, option, widget, returnData)


class OWAggregate(OWWidget, ConcurrentWidgetMixin):
    name = "Aggregate"
    description = ""
    icon = "icons/aggregate.svg"
    keywords = ["aggregate", "group by"]
    priority = 110

    class Inputs:
        data = Input("Data", Table, doc="Input data table")

    class Outputs:
        data = Output("Data", Table, doc="Input data table")

    class Warning(OWWidget.Warning):
        no_aggregations = Msg("Select at least one aggregation")
        cannot_compute = Msg("Some scores cannot be computed: {}")

    class Error(OWWidget.Error):
        row_col_intersection = Msg("Select different attributes for rows and columns")
        unexpected_error = Msg("{}")

    settingsHandler = DomainContextHandler()

    row_attrs: List[Variable] = ContextSetting([])
    col_attrs: List[Variable] = ContextSetting([])
    value_attrs: List[Variable] = ContextSetting([])
    x_variable: Variable = ContextSetting(None)

    aggregations: Set[str] = ContextSetting(set())
    freq_damping: bool = ContextSetting(False)
    freq_r2: bool = ContextSetting(False)
    custom_function: str = ContextSetting("")
    custom_function_name: str = ContextSetting("Custom function")
    auto_commit: bool = Setting(True)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)

        self.data = None
        self.result = None
        self.attrs_model = DomainModel(
            valid_types=(DiscreteVariable, ContinuousVariable),
            separators=False,
        )
        self.value_model = DomainModel(
            valid_types=(ContinuousVariable,), separators=False
        )

        self.__init_control_area()
        self.__init_main_area()

    def __init_control_area(self) -> None:
        box = gui.vBox(self.controlArea, "Rows (Group by)")
        self.row_attrs_view = AggregateListViewSearch(
            selectionMode=QListView.ExtendedSelection
        )
        self.row_attrs_view.setModel(self.attrs_model)
        self.row_attrs_view.selectionModel().selectionChanged.connect(
            self.__row_changed
        )
        box.layout().addWidget(self.row_attrs_view)

        box = gui.vBox(self.controlArea, "Columns (Split aggregation by)")
        self.col_attrs_view = ListViewSearch(selectionMode=QListView.ExtendedSelection)
        self.col_attrs_view.setModel(self.attrs_model)
        self.col_attrs_view.selectionModel().selectionChanged.connect(
            self.__col_changed
        )
        box.layout().addWidget(self.col_attrs_view)

        box = gui.vBox(self.controlArea, "Values to aggregate")
        self.val_attrs_view = AggregateListViewSearch(
            selectionMode=QListView.ExtendedSelection
        )
        self.val_attrs_view.setModel(self.value_model)
        self.val_attrs_view.selectionModel().selectionChanged.connect(
            self.__value_changed
        )
        box.layout().addWidget(self.val_attrs_view)

        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def __init_main_area(self) -> None:
        self.aggregation_cbs = {}
        self.additional_controls = defaultdict(list)

        def add_additional_settings(box, n):
            """Add additional controls bellow the checkbox"""
            in_box = gui.indentedBox(
                box,
                gui.checkButtonOffsetHint(self.aggregation_cbs[n]),
                orientation=Qt.Horizontal,
            )
            for p in AGG_PARAMS[n]:
                if isinstance(p, tuple):  # controls with values are in tuple
                    value, control = p
                    c = control(
                        widget=in_box,
                        master=self,
                        value=value,
                        callback=self.commit.deferred,
                    )
                    self.additional_controls[n].append(c)
                    # for checking the selection when start typing
                    if isinstance(c, QLineEdit):
                        c.textEdited.connect(partial(self.__set_cb_checked, n))
                    if value in AGGREGATION_INFO:
                        info = AGGREGATION_INFO[value]
                        in_box.layout().addWidget(InfoButton(info))
                        in_box.layout().addStretch(1)
                else:  # labels are strings
                    in_box.layout().addWidget(QLabel(p))
            self.__disable_sub_controls(n)

        def add_controls_to_box(box, aggs):
            """Add aggregations from aggregations dictionary to box"""
            box.layout().setSpacing(1)  # for smaller vertical spaces between cbs
            for n in aggs:
                label_text = n if n != "Custom function" else ""
                self.aggregation_cbs[n] = cb = QCheckBox(label_text)
                cb.setAttribute(Qt.WA_LayoutUsesWidgetRect)
                cb.stateChanged.connect(partial(self.__aggregation_changed, n))

                info = AGGREGATION_INFO.get(n)
                # hbox is required to add info button beside the checkbox
                b = gui.hBox(box) if info is not None else box
                cb.setSizePolicy(QSizePolicy.Fixed, cb.sizePolicy().verticalPolicy())
                b.layout().addWidget(cb)

                if n == "Custom function":
                    le = gui.lineEdit(
                        b,
                        self,
                        "custom_function_name",
                        callback=self.commit.deferred,
                        placeholderText="Name",
                    )
                    le.textEdited.connect(partial(self.__set_cb_checked, n))
                if info is not None:
                    b.layout().addWidget(InfoButton(info))
                    if n != "Custom function":
                        # make checkbox align left, it is not required for
                        # Custom function where lineEdit take full width
                        b.layout().addStretch(1)
                    b.layout().setSpacing(0)
                if n in AGG_PARAMS:
                    add_additional_settings(box, n)

        # create aggregations box
        self.agg_box = gui.vBox(self.mainArea, " ")
        add_controls_to_box(self.agg_box, AGGREGATIONS_BASIC)

        # create dependent aggregations box
        self.agg_dep_box = gui.vBox(self.mainArea, " ")
        hbox = gui.hBox(self.agg_dep_box)
        label = QLabel("x variable: ")
        label.setToolTip(X_ATTR_TOOLTIP)
        hbox.layout().addWidget(label)

        cb = gui.comboBox(
            hbox,
            self,
            "x_variable",
            model=self.value_model,
            searchable=True,
            callback=self.commit.deferred,
            tooltip=X_ATTR_TOOLTIP,
        )
        # make field to expand and label keep the text size
        cb.setSizePolicy(QSizePolicy.Expanding, cb.sizePolicy().verticalPolicy())
        add_controls_to_box(self.agg_dep_box, AGGREGATIONS_DEP)

        gui.rubber(self.mainArea)

    @staticmethod
    def _get_selection(view: QListView) -> List[Any]:
        rows = view.selectionModel().selectedRows()
        values = view.model()[:]
        return [values[row.row()] for row in sorted(rows)]

    @staticmethod
    def _set_selection(view: QListView, selected: List[Any], cb: Callable) -> None:
        sm = view.selectionModel()
        values = view.model()[:]
        with disconnected(sm.selectionChanged, cb):
            for val in selected:
                index = values.index(val)
                model_index = view.model().index(index, 0)
                sm.select(model_index, QItemSelectionModel.Select)

    def __set_cb_checked(self, key: str) -> None:
        """
        This function is called when user changes text in the control besides
        the checkbox. Function check matching checkbox if not checked.
        StateChange signal must not be emitted, since it will be emitted when
        user finises with editing of the corresponding control.
        """
        cb = self.aggregation_cbs[key]
        cb.blockSignals(True)
        cb.setChecked(True)
        cb.blockSignals(False)
        # since signals are blocked aggregation must be added to set manually
        self.aggregations.add(key)

    def __row_changed(self) -> None:
        self.row_attrs = self._get_selection(self.row_attrs_view)
        self.commit.deferred()

    def __col_changed(self) -> None:
        self.col_attrs = self._get_selection(self.col_attrs_view)
        self.commit.deferred()

    def __value_changed(self) -> None:
        self.value_attrs = self._get_selection(self.val_attrs_view)
        self.commit.deferred()

    def __aggregation_changed(self, key: str, state: int) -> None:
        if Qt.CheckState(state) == Qt.Unchecked:
            self.aggregations.discard(key)
        else:
            self.aggregations.add(key)
        self.__disable_sub_controls(key)
        self.commit.deferred()

    def __disable_sub_controls(self, key):
        """Disable the sub-checkboxes for specific aggregation"""
        disabling_supported = (QCheckBox,)
        for c in self.additional_controls.get(key, []):
            if isinstance(c, disabling_supported):
                c.setEnabled(key in self.aggregations)

    @Inputs.data
    def set_data(self, data: Table) -> None:
        self.closeContext()
        self.data = data
        self.cancel()
        self.result = Result()
        self.Outputs.data.send(None)

        self.__set_up_models()
        self.openContext(self.data)
        if self.x_variable is None and len(self.value_model) > 0:
            self.x_variable = self.value_model[0]
        self.__setup_controls_selection()
        if data:
            self.commit.now()

    def __set_up_models(self) -> None:
        self.attrs_model.set_domain(self.data.domain if self.data else None)
        self.value_model.set_domain(self.data.domain if self.data else None)
        if self.data:
            self.row_attrs = self.attrs_model[:1]
            attrs = iter([x for x in self.value_model if x != self.row_attrs[0]])
            val_attr = next(attrs, None)
            self.value_attrs = [val_attr] if val_attr else []
            self.x_variable = next(
                attrs, self.value_model[0] if len(self.value_model) > 0 else None
            )
        else:
            self.row_attrs, self.col_attrs, self.value_attrs = [], [], []
            self.x_variable = None
        self.col_attrs = []
        self.aggregations = {next(iter(AGGREGATIONS_BASIC))}

    @gui.deferred
    def commit(self) -> None:
        self.Error.clear()
        self.Warning.clear()
        if self.data is None:
            return
        if len(set(self.row_attrs) & set(self.col_attrs)) > 0:
            self.Error.row_col_intersection()
            self.Outputs.data.send(None)
        else:
            self.__aggregate()

    def __aggregate(self) -> None:
        def not_empty_string(params_):
            return all(s != "" for s in params_)

        aggregations = []
        for a, fun in chain(AGGREGATIONS_BASIC.items(), AGGREGATIONS_DEP.items()):
            if a in self.aggregations:
                if a in AGG_PARAMS:
                    params = tuple(
                        getattr(self, p[0])
                        for p in AGG_PARAMS[a]
                        if isinstance(p, tuple)
                    )
                    if a == "Custom function":
                        params = (self.custom_function_name,) + params
                    if not_empty_string(params):  # if fields not empty
                        aggregations.append(((a,) + params, partial(fun, *params)))
                else:
                    aggregations.append((a, fun))
        if aggregations and self.value_attrs:
            self.start(
                _run,
                self.data,
                self.row_attrs,
                self.col_attrs,
                self.value_attrs,
                aggregations,
                self.x_variable,
                self.result,
            )
        else:
            if not aggregations:
                self.Warning.no_aggregations()
            self.Outputs.data.send(None)

    def on_done(self, result: Result) -> None:
        self.result = result
        if result.warnings:
            self.Warning.cannot_compute(", ".join(result.warnings))
        self.Outputs.data.send(result.result_table)

    def on_partial_result(self, _: Any) -> None:
        pass

    def on_exception(self, ex: Exception) -> None:
        self.Error.unexpected_error(str(ex))

    def sizeHint(self) -> QSize:
        return QSize(600, 500)

    def __setup_controls_selection(self) -> None:
        for view, attr, cb in [
            (self.row_attrs_view, self.row_attrs, self.__row_changed),
            (self.col_attrs_view, self.col_attrs, self.__col_changed),
            (self.val_attrs_view, self.value_attrs, self.__value_changed),
        ]:
            self._set_selection(view, attr, cb)

        for key, cb in self.aggregation_cbs.items():
            cb.blockSignals(True)
            cb.setChecked(key in self.aggregations)
            cb.blockSignals(False)
            self.__disable_sub_controls(key)


class AggregatePreprocessor(Transformation):
    def __init__(
        self,
        row_attrs: List[Variable],
        col_attrs: List[Variable],
        val_attrs: List[Variable],
        aggregations: List[Tuple[str, Union[str, Callable]]],
        x_attr: Optional[Variable],
    ):
        super().__init__()
        self.row_attrs = row_attrs
        self.col_attrs = col_attrs
        self.val_attrs = val_attrs
        self.aggregations = aggregations
        self.x_attr = x_attr
        # domain after transformation made by widget
        # to know if any Orange transformation happened after
        self.domain = None

    def __call__(self, data: Table) -> Table:
        self.check_attributes_in_data(data)
        results = _run(
            data,
            self.row_attrs,
            self.col_attrs,
            self.val_attrs,
            self.aggregations,
            self.x_attr,
            Result(),
        )
        return results.result_table

    def check_attributes_in_data(self, data):
        all_attrs = self.row_attrs + self.col_attrs + self.val_attrs + [self.x_attr]
        missing = sorted([var.name for var in all_attrs if var not in data.domain])
        if missing:
            raise ValueError(f"Data missing attributes: {', '.join(missing)}")

    def __repr__(self):
        aggrs = (x if isinstance(x, str) else x[0] for x, _ in self.aggregations)
        col_attr = map(str, self.col_attrs) if self.col_attrs else ["N/A"]
        table = (
            ("Rows (Group by)", ", ".join(map(str, self.row_attrs))),
            ("Columns (Split by)", ", ".join(col_attr)),
            ("Values to aggregate", ", ".join(map(str, self.val_attrs))),
            ("Aggregations", ", ".join(aggrs)),
            ("X variable", self.x_attr.name if self.x_attr else "N/A"),
        )
        return f"<h4>Aggregate</h4>{create_info_html_table(table)}"


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview

    WidgetPreview(OWAggregate).run(Table("heart_disease"))
