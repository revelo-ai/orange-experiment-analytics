from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, Any
from types import SimpleNamespace
from xml.sax.saxutils import escape
import os
from functools import singledispatch

import numpy as np
import pandas as pd

from AnyQt.QtCore import Qt, QSize, QItemSelection, QItemSelectionModel, \
    QItemSelectionRange, QPointF, Signal, QRectF, QDateTime, QTimeZone
from AnyQt.QtGui import QColor, QPainter, QTransform
from AnyQt.QtWidgets import QSizePolicy, QWidget, QGridLayout, QLabel, \
    QLineEdit, QVBoxLayout, QPushButton, QDoubleSpinBox, QScrollArea, \
    QDateTimeEdit

import pyqtgraph as pg
from Orange.preprocess import Preprocess
from pyqtgraph.graphicsItems.ViewBox import ViewBox

from orangewidget.utils.listview import ListViewSearch
from orangewidget.utils.visual_settings_dlg import VisualSettingsDialog

from Orange.data import Table, DiscreteVariable, ContinuousVariable, \
    FilterContinuous, Values, Variable, TimeVariable
from Orange.data.util import get_unique_names
from Orange.widgets import gui, report
from Orange.widgets.settings import Setting, ContextSetting, \
    DomainContextHandler
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.utils.itemmodels import DomainModel
from Orange.widgets.utils.plot import OWPlotGUI, SELECT, PANNING, ZOOMING
from Orange.widgets.utils.sql import check_sql_input
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.visualize.owdistributions import LegendItem
from Orange.widgets.visualize.owscatterplotgraph import AxisItem
from Orange.widgets.visualize.utils.customizableplot import Updater, \
    CommonParameterSetter
from Orange.widgets.visualize.utils.plotutils import PlotWidget
from Orange.widgets.widget import OWWidget, Input, Output, Msg

from orangecontrib.experiment_analytics.transformation_export import (
    add_transformation_to_data,
    Transformation,
    create_info_html_table,
)

SpinType = Union[QDoubleSpinBox, QDateTimeEdit]


@contextmanager
def disconnected(signal, slot, type=Qt.UniqueConnection):
    signal.disconnect(slot)
    try:
        yield
    finally:
        signal.connect(slot, type)


class Group:
    color: QColor = None
    line: Tuple[np.ndarray, np.ndarray, np.ndarray] = None
    rnge: Tuple[np.ndarray, np.ndarray, np.ndarray] = None
    mean: Tuple[np.ndarray, np.ndarray] = None


class Result(SimpleNamespace):
    x_bounds: Tuple[float, float] = None
    y_bounds: Tuple[float, float] = None
    groups: List[Group] = None


class Runner:
    def run(cls, x_data: np.ndarray, y_data: np.ndarray,
            keys: List[np.array], group_data: Optional[np.ndarray],
            group_var: Optional[DiscreteVariable], state: TaskState) -> Result:

        state.set_status("Plotting...")

        result = Result()
        result.x_bounds = np.nanmin(x_data), np.nanmax(x_data)
        result.y_bounds = np.nanmin(y_data), np.nanmax(y_data)

        if not keys:
            keys = [np.zeros(len(x_data))]

        key = np.vstack(keys).T
        if state.is_interruption_requested():
            raise Exception

        state.set_progress_value(1)
        cls._check_valid_series(x_data, key)
        state.set_progress_value(10)
        if state.is_interruption_requested():
            raise Exception

        groups = []
        if not group_var:
            group = cls._get_group(
                x_data, y_data, key, QColor(Qt.darkGray), state
            )
            groups.append(group)
        else:
            assert group_data is not None
            unique_data = np.unique(group_data)
            steps = len(unique_data)
            for i in unique_data:
                state.set_progress_value(10 + 90 * i / steps)
                mask = group_data == i
                group = cls._get_group(
                    x_data[mask], y_data[mask], key[mask],
                    QColor(*group_var.colors[int(i)]), state
                )
                groups.append(group)
        result.groups = groups

        return result

    @classmethod
    def _get_group(
            cls,
            x_data: np.ndarray,
            y_data: np.ndarray,
            key: np.ndarray,
            color: QColor,
            state: TaskState
    ) -> Group:
        if state.is_interruption_requested():
            raise Exception

        assert not any(np.isnan(x_data))
        assert not any(np.isnan(y_data))

        if state.is_interruption_requested():
            raise Exception

        df = pd.DataFrame(np.vstack([x_data, y_data]).T)
        func = ["min", "max", "mean"]
        agg = df.groupby(0).aggregate(func).reset_index().values

        if state.is_interruption_requested():
            raise Exception

        group = Group()
        group.color = color
        group.line = cls.__get_line(x_data, y_data, key, state)
        group.rnge = agg[:, 0], agg[:, 1], agg[:, 2]
        group.mean = agg[:, 0], agg[:, 3]
        return group

    @staticmethod
    def __get_line(
            x_data: np.ndarray,
            y_data: np.ndarray,
            key: np.ndarray,
            state: TaskState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = [x_data] + [key[:, i] for i in range(key.shape[1])]
        if state.is_interruption_requested():
            raise Exception
        indices = np.lexsort(arrays)
        if state.is_interruption_requested():
            raise Exception
        con = np.zeros(len(x_data), dtype=bool)
        con[:-1] = np.all(key[indices][:-1] == key[indices][1:], axis=1)
        if state.is_interruption_requested():
            raise Exception
        return x_data[indices], y_data[indices], con

    @staticmethod
    def _check_valid_series(x_data: np.ndarray, key: np.ndarray) -> None:
        stacked = np.hstack([key, x_data[:, None]])
        if len(np.unique(stacked, axis=0)) != len(stacked):
            raise NonUniqueSeries()


class SelectionRect(pg.GraphicsObject):
    def __init__(self, rect: QRectF):
        self.__rect: QRectF = rect
        super().__init__()

    def boundingRect(self) -> QRectF:
        return self.__rect

    def paint(self, painter: QPainter, *_):
        painter.save()
        painter.setPen(pg.mkPen((255, 255, 0), width=1))
        painter.setBrush(pg.mkBrush(255, 255, 0, 100))
        painter.drawRect(self.__rect)
        painter.restore()


class SlicerPlotViewBox(ViewBox):
    sigSelectionChanged = Signal(QPointF, QPointF)
    sigDeselect = Signal()

    def __init__(self, graph: pg.PlotWidget):
        super().__init__(enableMenu=False)
        self.__graph = graph
        self.__graph_state: int = None
        self.set_graph_state(SELECT)

        self.__selection_rect = SelectionRect(QRectF(0, 0, 1, 1))
        self.__selection_rect.setZValue(1e9)
        self.__selection_rect.hide()
        self.addItem(self.__selection_rect)

    def set_graph_state(self, state: int):
        modes = {SELECT: self.RectMode,
                 PANNING: self.PanMode,
                 ZOOMING: self.RectMode}
        self.__graph_state = state
        self.setMouseMode(modes[state])

    def mouseDragEvent(self, ev, axis=None):
        if self.__graph_state == SELECT and axis is None:
            ev.accept()

            if ev.button() == Qt.LeftButton and self.__graph.y_bounds:
                p1, p2 = ev.buttonDownPos(), ev.pos()
                p1, p2 = self.mapToView(p1), self.mapToView(p2)
                p1.setY(self.__graph.y_bounds[0])
                p2.setY(self.__graph.y_bounds[1])
                x1, x2 = self.__graph.x_bounds
                p1.setX(max(min(p1.x(), x2), x1))
                p2.setX(max(min(p2.x(), x2), x1))

                if ev.isFinish():
                    self.__selection_rect.hide()
                    self.sigSelectionChanged.emit(p1, p2)
                else:
                    self._update_selection_rect(p1, p2)

        elif self.__graph_state == ZOOMING or self.__graph_state == PANNING:
            ev.ignore()
            super().mouseDragEvent(ev, axis=axis)
        else:
            ev.ignore()

    def _update_selection_rect(self, p1: QPointF, p2: QPointF):
        rect = QRectF(p1, p2)
        self.__selection_rect.setPos(rect.topLeft())
        trans = QTransform.fromScale(rect.width(), rect.height())
        self.__selection_rect.setTransform(trans)
        self.__selection_rect.show()

    def mouseClickEvent(self, ev):
        ev.accept()
        self.sigDeselect.emit()


class NonUniqueSeries(Exception):
    pass


class PlotParameterSetter(CommonParameterSetter):
    MEAN_LABEL = "Mean"
    LINE_LABEL = "Lines"
    RANGE_LABEL = "Range"

    LINE_WIDTH = 1
    LINE_ALPHA = 200
    RANGE_ALPHA = 25

    MEAN_WIDTH = 6
    MEAN_DARK_FACTOR = 110

    def __init__(self, master):
        self.master: SlicerPlot = master
        self.mean_settings = {
            Updater.WIDTH_LABEL: self.MEAN_WIDTH,
            Updater.STYLE_LABEL: Updater.DEFAULT_LINE_STYLE,
        }
        self.line_settings = {
            Updater.WIDTH_LABEL: self.LINE_WIDTH,
            Updater.ALPHA_LABEL: self.LINE_ALPHA,
            Updater.STYLE_LABEL: Updater.DEFAULT_LINE_STYLE,
            Updater.ANTIALIAS_LABEL: True,
        }
        self.range_settings = {
            Updater.ALPHA_LABEL: self.RANGE_ALPHA,
        }
        super().__init__()

    def update_setters(self):
        self.initial_settings = {
            self.LABELS_BOX: {
                self.FONT_FAMILY_LABEL: self.FONT_FAMILY_SETTING,
                self.TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TITLE_LABEL: self.FONT_SETTING,
                self.AXIS_TICKS_LABEL: self.FONT_SETTING,
                self.LEGEND_LABEL: self.FONT_SETTING,
            },
            self.ANNOT_BOX: {
                self.TITLE_LABEL: {self.TITLE_LABEL: ("", "")},
            },
            self.PLOT_BOX: {
                self.MEAN_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15), self.MEAN_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          Updater.DEFAULT_LINE_STYLE),
                },
                self.LINE_LABEL: {
                    Updater.WIDTH_LABEL: (range(1, 15), self.LINE_WIDTH),
                    Updater.STYLE_LABEL: (list(Updater.LINE_STYLES),
                                          Updater.DEFAULT_LINE_STYLE),
                    Updater.ALPHA_LABEL: (range(0, 255, 5), self.LINE_ALPHA),
                    Updater.ANTIALIAS_LABEL: (None, True),
                },
                self.RANGE_LABEL: {
                    Updater.ALPHA_LABEL: (range(0, 255, 5),
                                          self.RANGE_ALPHA),
                },
            }
        }

        def update_mean(**settings):
            self.mean_settings.update(**settings)
            Updater.update_lines(self.master.mean_items, **self.mean_settings)

        def update_lines(**settings):
            self.line_settings.update(**settings)
            Updater.update_lines(self.master.line_items, **self.line_settings)

        def _update_brush(items, **settings):
            for item in items:
                brush = item.brush()
                color = brush.color()
                color.setAlpha(settings[Updater.ALPHA_LABEL])
                brush.setColor(color)
                item.setBrush(brush)

        def update_range(**settings):
            self.range_settings.update(**settings)
            _update_brush(self.master.range_items, **settings)

        self._setters[self.PLOT_BOX] = {
            self.MEAN_LABEL: update_mean,
            self.LINE_LABEL: update_lines,
            self.RANGE_LABEL: update_range,
        }

    @property
    def title_item(self):
        return self.master.getPlotItem().titleLabel

    @property
    def axis_items(self):
        return [value["item"] for value in
                self.master.getPlotItem().axes.values()]

    @property
    def legend_items(self):
        return self.master.legend.items

    @property
    def getAxis(self):
        return self.master.getAxis


class SlicerPlot(PlotWidget):
    selectionChanged = Signal(list)

    def __init__(self, parent: OWWidget, show_lines: bool,
                 show_range: bool, show_mean: bool):

        # data
        self.__x_bounds: Tuple[float, float] = None
        self.__y_bounds: Tuple[float, float] = None

        # settings
        self.__show_lines = show_lines
        self.__show_range = show_range
        self.__show_mean = show_mean

        # items
        self.__line_items: List[pg.PlotCurveItem] = []
        self.__range_items: List[pg.FillBetweenItem] = []
        self.__mean_items: List[pg.PlotCurveItem] = []

        # selection
        self.__selection: List = []
        self.__selection_rect_items: List[SelectionRect] = []

        self._view_box = SlicerPlotViewBox(self)
        self._view_box.sigSelectionChanged.connect(self._update_selection)
        self._view_box.sigDeselect.connect(self._deselect)
        super().__init__(parent, viewBox=self._view_box, enableMenu=False,
                         axisItems={"bottom": AxisItem(orientation="bottom"),
                                    "left": AxisItem(orientation="left")})
        self.hideButtons()
        self.getPlotItem().setContentsMargins(10, 10, 10, 10)

        self._set_range()
        self.__legend: LegendItem = self._create_legend()

        self.plot_setter = PlotParameterSetter(self)

    @property
    def x_bounds(self) -> Tuple[float, float]:
        return self.__x_bounds

    @property
    def y_bounds(self) -> Tuple[float, float]:
        return self.__y_bounds

    @property
    def line_items(self) -> List[pg.PlotCurveItem]:
        return self.__line_items

    @property
    def range_items(self) -> List[pg.FillBetweenItem]:
        return self.__range_items

    @property
    def mean_items(self) -> List[pg.PlotCurveItem]:
        return self.__mean_items

    @property
    def legend(self) -> pg.LegendItem:
        return self.__legend

    def clear_plot(self):
        self.clear()
        self.__x_bounds = None
        self.__y_bounds = None

        for i in range(len(self.__line_items)):
            self.removeItem(self.__line_items[i])
            self.removeItem(self.__range_items[i])
            self.removeItem(self.__mean_items[i])
        self.__line_items.clear()
        self.__range_items.clear()
        self.__mean_items.clear()

        self._clear_selection()

        self.__legend.clear()
        self.__legend.hide()

    def _clear_selection(self):
        self.__selection = []
        for i in range(len(self.__selection_rect_items)):
            self.removeItem(self.__selection_rect_items[i])
        self.__selection_rect_items.clear()

    def set_data(self, result: Result, x_var: ContinuousVariable,
                 y_var: ContinuousVariable,
                 group_var: Optional[DiscreteVariable]):
        self.__x_bounds = result.x_bounds
        self.__y_bounds = result.y_bounds
        self._set_range()
        self._set_axes(x_var, y_var)
        self._set_legend(group_var)
        self._plot_data(result.groups)

    def _set_axes(self, x_var: ContinuousVariable, y_var: ContinuousVariable):
        bottom_axis: AxisItem = self.getAxis("bottom")
        bottom_axis.setLabel(x_var.name)
        bottom_axis.use_time(x_var.is_time)

        left_axis: AxisItem = self.getAxis("left")
        left_axis.setLabel(y_var.name)
        left_axis.use_time(y_var.is_time)

    def _set_legend(self, group_var: Optional[DiscreteVariable]):
        if not group_var:
            return

        assert isinstance(group_var, DiscreteVariable)

        for name, color in zip(group_var.values, group_var.colors):
            c = QColor(*color)
            dots = pg.ScatterPlotItem(pen=c, brush=c, size=10, shape="o")
            self.__legend.addItem(dots, escape(name))
        self.__legend.show()
        Updater.update_legend_font(self.plot_setter.legend_items,
                                   **self.plot_setter.legend_settings)

    def _plot_data(self, groups: List[Group]):
        for group in groups:
            self._add_group_items(group)

    def _add_group_items(self, group: Group):
        x_data, y_data, con = group.line
        assert not any(np.isnan(x_data))
        assert not any(np.isnan(y_data))
        self.__add_line_item(x_data, y_data, con, group.color)
        x_data, y_bottom, y_top = group.rnge
        self.__add_range_item(x_data, y_bottom, y_top, group.color)
        x_data, y_data = group.mean
        self.__add_mean_item(x_data, y_data, group.color)

    def __add_line_item(self, x_data: np.ndarray, y_data: np.ndarray,
                        con: np.ndarray, color: QColor):
        line = pg.PlotCurveItem(x=x_data, y=y_data,
                                connect=con, pen=pg.mkPen(color))
        line.setVisible(self.__show_lines)
        Updater.update_lines([line], **self.plot_setter.line_settings)
        self.addItem(line)
        self.__line_items.append(line)

    def __add_range_item(self, x_data: np.ndarray, y_bottom: np.ndarray,
                         y_top: np.ndarray, color: QColor):
        alpha = self.plot_setter.range_settings[Updater.ALPHA_LABEL]
        lighter_color = QColor(color)
        lighter_color.setAlpha(alpha)
        fill = pg.FillBetweenItem(
            pg.PlotDataItem(x=x_data, y=y_bottom),
            pg.PlotDataItem(x=x_data, y=y_top),
            brush=pg.mkBrush(lighter_color)
        )
        fill.setVisible(self.__show_range)
        self.addItem(fill)
        self.__range_items.append(fill)

    def __add_mean_item(self, x_data: np.ndarray, y_data: np.ndarray,
                        color: QColor):
        line = pg.PlotCurveItem(x=x_data, y=y_data, pen=pg.mkPen(color))
        line.setVisible(self.__show_mean)
        Updater.update_lines([line], **self.plot_setter.mean_settings)
        self.addItem(line)
        self.__mean_items.append(line)

    def set_show_lines(self, show: bool):
        if self.__show_lines != show:
            self.__show_lines = show
            for line in self.__line_items:
                line.setVisible(show)

    def set_show_range(self, show: bool):
        if self.__show_range != show:
            self.__show_range = show
            for fill in self.__range_items:
                fill.setVisible(show)

    def set_show_mean(self, show: bool):
        if self.__show_mean != show:
            self.__show_mean = show
            for line in self.__mean_items:
                line.setVisible(show)

    def apply_selection(self, selection: List[Tuple[float, float]]):
        self._clear_selection()
        self.__selection = selection
        for left, right in selection:
            p1 = QPointF(left, self.__y_bounds[1])
            p2 = QPointF(right, self.__y_bounds[0])
            sel_rect_item = SelectionRect(QRectF(p1, p2))
            self.addItem(sel_rect_item)
            self.__selection_rect_items.append(sel_rect_item)

    def _update_selection(self, p1: QPointF, p2: QPointF):
        rect = QRectF(p1, p2).normalized()
        self.__selection.append((rect.topLeft().x(), rect.topRight().x()))

        sel_rect_item = SelectionRect(rect)
        self.addItem(sel_rect_item)
        self.__selection_rect_items.append(sel_rect_item)
        self.selectionChanged.emit(self.__selection)

    def _deselect(self):
        for i in range(len(self.__selection_rect_items)):
            self.removeItem(self.__selection_rect_items[i])
        self.__selection_rect_items.clear()

        if self.__selection:
            self.__selection = []
            self.selectionChanged.emit(self.__selection)

    def select_button_clicked(self):
        self._view_box.set_graph_state(SELECT)

    def pan_button_clicked(self):
        self._view_box.set_graph_state(PANNING)

    def zoom_button_clicked(self):
        self._view_box.set_graph_state(ZOOMING)

    def reset_button_clicked(self):
        self._set_range()

    def _set_range(self):
        x_range = self.__x_bounds or (0, 1)
        y_range = self.__y_bounds or (0, 1)
        self._view_box.setRange(xRange=x_range, yRange=y_range, padding=0.03)

    def _create_legend(self) -> LegendItem:
        legend = LegendItem()
        legend.setParentItem(self._view_box)
        legend.anchor((1, 0), (1, 0))
        legend.hide()
        return legend


class SlicePicker(QWidget):
    REMOVE, FROM, LABEL, TO, NAME = range(5)
    sigGeometryChanged = Signal()
    sigSlicesChanged = Signal(list)

    def __init__(self, parent: QWidget):
        super().__init__(parent)

        # data
        self.__data: List[List[List[float], str]] = []

        # default spin settings
        self.__min: float = None
        self.__max: float = None
        self.__x_var: Variable = None

        # controls
        self.__slices_box: QGridLayout = None
        self.__add_button: QPushButton = None
        self.__controls: List[Tuple[QPushButton, SpinType, QLabel,
                                    SpinType, QLineEdit]] = []
        self._setup_gui()

    @property
    def _remove_buttons(self) -> List[QPushButton]:
        return [controls[self.REMOVE] for controls in self.__controls]

    @property
    def _start_spins(self) -> List[QDoubleSpinBox]:
        return [controls[self.FROM] for controls in self.__controls]

    @property
    def _end_spins(self) -> List[QDoubleSpinBox]:
        return [controls[self.TO] for controls in self.__controls]

    @property
    def _name_edits(self) -> List[QLineEdit]:
        return [controls[self.NAME] for controls in self.__controls]

    @property
    def _use_date_time(self) -> bool:
        return isinstance(self.__x_var, TimeVariable) and \
               (self.__x_var.have_date or self.__x_var.have_time)

    def _setup_gui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)

        box = gui.vBox(self, box=False)
        self.__slices_box = QGridLayout()
        box.layout().addLayout(self.__slices_box)
        button_box = gui.hBox(box)
        self.__add_button = gui.button(
            button_box, self, "+", callback=self.__on_add_button_clicked,
            width=34, autoDefault=False, enabled=False,
            sizePolicy=(QSizePolicy.Maximum, QSizePolicy.Maximum)
        )
        gui.rubber(button_box)

    def __on_add_button_clicked(self):
        self._add_row()
        self.sigSlicesChanged.emit(self.__data)

    def _add_row(self, start_value=None, end_value=None, label=None):
        assert self.__min is not None
        assert self.__max is not None
        assert self.__x_var is not None

        start_spin, end_spin = \
            self.__get_range_controls(start_value, end_value)

        row_id = len(self.__controls)
        edit = QLineEdit(text=f"Slice {row_id + 1}")
        edit.setFixedWidth(90)
        if label is not None:
            edit.setText(label)
        edit.textChanged.connect(self.__on_text_changed)

        button = gui.button(
            None, self, "×", callback=self.__on_remove_button_clicked,
            width=12, height=20, autoDefault=False, flat=True,
            styleSheet="* {font-size: 16pt; color: silver}"
                       "*:hover {color: black}"
        )

        controls = (button, start_spin, QLabel("to"), end_spin, edit)
        n_rows = self.__slices_box.rowCount()
        for i, control in enumerate(controls):
            self.__slices_box.addWidget(control, n_rows, i)
        if row_id == 0:
            self.sigGeometryChanged.emit()

        self.__controls.append(controls)
        if self._use_date_time:
            min_max = [start_spin.dateTime().toSecsSinceEpoch(),
                       end_spin.dateTime().toSecsSinceEpoch()]
        else:
            min_max = [start_spin.value(), end_spin.value()]
        self.__data.append([min_max, edit.text()])

    def __get_range_controls(self, start_value, end_value) -> \
            Tuple[SpinType, SpinType]:

        TIME_FORMAT = "hh:mm:ss"
        DATE_FORMAT = "yyyy.MM.dd"

        if self._use_date_time:
            min_ = QDateTime.fromSecsSinceEpoch(
                int(self.__min), QTimeZone.utc())
            max_ = QDateTime.fromSecsSinceEpoch(
                int(self.__max), QTimeZone.utc())
            if not self.__x_var.have_date and self.__x_var.have_time:
                format_ = TIME_FORMAT
            elif self.__x_var.have_date and not self.__x_var.have_time:
                format_ = DATE_FORMAT
            else:
                format_ = f"{DATE_FORMAT} {TIME_FORMAT}"
            kwargs = {"minimumDateTime": min_, "maximumDateTime": max_,
                      "timeSpec": Qt.UTC, "displayFormat": format_}

            start_spin = QDateTimeEdit(**kwargs)
            end_spin = QDateTimeEdit(**kwargs)

            start_callback = start_spin.dateTimeChanged
            end_callback = end_spin.dateTimeChanged

        else:
            n_dec = min(self.__x_var.number_of_decimals + 1, 3)
            kwargs = {"minimum": self.__min, "maximum": self.__max,
                      "decimals": n_dec, "singleStep": 10 ** (-n_dec)}

            start_spin = QDoubleSpinBox(**kwargs)
            end_spin = QDoubleSpinBox(**kwargs)

            start_callback = start_spin.valueChanged
            end_callback = end_spin.valueChanged

        if start_value is not None:
            _set_value(start_spin, start_value)
            _set_min(start_spin, end_spin)
        else:
            _set_value(start_spin, self.__min)
        _set_value(end_spin, self.__min if end_value is None else end_value)

        start_callback.connect(self.__on_spin_start_changed)
        end_callback.connect(self.__on_spin_end_changed)

        return start_spin, end_spin

    def __on_spin_start_changed(self):
        start_spin: SpinType = self.sender()
        row_id = self._start_spins.index(start_spin)
        end_spin = self._end_spins[row_id]
        _set_min(start_spin, end_spin)
        self.__data[row_id][0][0] = _get_value(start_spin)
        self.sigSlicesChanged.emit(self.__data)

    def __on_spin_end_changed(self):
        end_spin: SpinType = self.sender()
        row_id = self._end_spins.index(end_spin)
        self.__data[row_id][0][1] = _get_value(end_spin)
        self.sigSlicesChanged.emit(self.__data)

    def __on_text_changed(self):
        line_edit: QLineEdit = self.sender()
        row_id = self._name_edits.index(line_edit)
        self.__data[row_id][1] = line_edit.text()
        self.sigSlicesChanged.emit(self.__data)

    def __on_remove_button_clicked(self):
        index = self._remove_buttons.index(self.sender())
        self._remove_row(index)
        self.sigSlicesChanged.emit(self.__data)

    def _remove_row(self, row_index: int):
        assert len(self.__controls) > row_index
        for col_index in range(len(self.__controls[row_index])):
            widget = self.__controls[row_index][col_index]
            if widget is not None:
                self.__slices_box.removeWidget(widget)
                widget.deleteLater()
        del self.__controls[row_index]
        del self.__data[row_index]

    def _remove_rows(self):
        for row in range(len(self.__controls) - 1, -1, -1):
            self._remove_row(row)
        self.__data.clear()
        self.__controls.clear()

    def clear_all(self):
        self.__min = None
        self.__max = None
        self._remove_rows()

    def set_parameters(self, minimum: float, maximum: float,
                       x_var: ContinuousVariable):
        self.__min = minimum
        self.__max = maximum
        self.__x_var = x_var

    def set_data(self, data: List[Tuple[Tuple[float, float], str]]):
        self._remove_rows()
        for min_max, label in data:
            self._add_row(min_max[0], min_max[1], label)

    def set_add_enabled(self, enable: bool):
        self.__add_button.setEnabled(enable)


@singledispatch
def _get_value(_) -> float:
    raise NotImplementedError


@_get_value.register(QDoubleSpinBox)
def _(control: QDoubleSpinBox) -> float:
    return control.value()


@_get_value.register(QDateTimeEdit)
def _(control: QDateTimeEdit) -> float:
    return control.dateTime().toSecsSinceEpoch()


@singledispatch
def _set_value(*_):
    raise NotImplementedError


@_set_value.register(QDoubleSpinBox)
def _(control: QDoubleSpinBox, value: float):
    return control.setValue(value)


@_set_value.register(QDateTimeEdit)
def _(control: QDateTimeEdit, value: float):
    date_time = QDateTime.fromSecsSinceEpoch(int(value), QTimeZone.utc())
    return control.setDateTime(date_time)


@singledispatch
def _set_min(*_):
    raise NotImplementedError


@_set_min.register(QDoubleSpinBox)
def _(start: QDoubleSpinBox, end: QDoubleSpinBox):
    end.setMinimum(max(end.minimum(), start.value()))


@_set_min.register(QDateTimeEdit)
def _(start: QDateTimeEdit, end: QDateTimeEdit):
    minimum = max(end.minimumDateTime(), start.dateTime())
    end.setMinimumDateTime(minimum)


def _create_output_table(data, x_var, selection) -> Table:
    assert data is not None
    assert x_var is not None

    if not selection:
        # when no selection return original data
        return data.copy()

    unique_name = get_unique_names(data.domain, "Slice")
    slice_var_values = tuple(n for _, n in selection)
    slice_var = DiscreteVariable(unique_name, values=slice_var_values)

    tables = []
    for rng, name in selection:
        f = FilterContinuous(x_var, FilterContinuous.Between, *rng)
        table = Values([f])(data)
        table = table.add_column(
            slice_var, np.full(len(table), slice_var_values.index(name)), to_metas=True
        )
        tables.append(table)

    return Table.concatenate(tables, axis=0)


class OWSlicer(OWWidget, ConcurrentWidgetMixin):
    name = "Series Slicer"
    description = "Visualization and selection of time series data."
    icon = "icons/slicer.svg"
    priority = 100
    keywords = ["time", "series", "slice", "line", "chart", "plot"]

    buttons_area_orientation = Qt.Vertical

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        selected_data = Output("Selected Data", Table)

    class Error(OWWidget.Error):
        no_valid_data = Msg("No plot due to no valid data.")
        non_unique_series = Msg("Non-unique series values.\n"
                                "Use 'Split by' to define a unique series.")
        domain_duplicates = Msg("Slice labels should be unique.")

    class Warning(OWWidget.Warning):
        no_continuous_vars = Msg("Data has no numeric variables.")
        no_display_option = Msg("No display option is selected.")

    class Information(OWWidget.Information):
        hidden_instances = Msg("Instances with unknown values are not shown.")

    settingsHandler = DomainContextHandler()
    x_var: Optional[ContinuousVariable] = ContextSetting(None)
    y_var: Optional[ContinuousVariable] = ContextSetting(None)
    key_vars: List[Variable] = ContextSetting([])
    group_var: Optional[DiscreteVariable] = ContextSetting(None)
    show_profiles = Setting(False)
    show_range = Setting(True)
    show_mean = Setting(True)
    auto_commit = Setting(True)
    selection: Optional[List[List[Any]]] = Setting(None, schema_only=True)
    visual_settings = Setting({}, schema_only=True)

    graph_name = "graph.plotItem"

    def __init__(self):
        OWWidget.__init__(self)
        ConcurrentWidgetMixin.__init__(self)

        self.data: Optional[Table] = None
        self.graph: SlicerPlot = None

        self._key_vars_view: ListViewSearch = None
        self._xy_vars_model = DomainModel(DomainModel.MIXED,
                                          valid_types=ContinuousVariable)
        self._key_vars_model = DomainModel(separators=False,
                                           valid_types=(DiscreteVariable,
                                                        ContinuousVariable))
        self._group_var_model = DomainModel(placeholder="None",
                                            separators=False,
                                            valid_types=DiscreteVariable)
        self._slice_picker: SlicePicker = None

        self.__pending_selection: \
            Optional[List[Tuple[Tuple[float, float], str]]] = self.selection

        self.setup_gui()
        VisualSettingsDialog(self, self.graph.plot_setter.initial_settings)

    def setup_gui(self):
        self._add_graph()
        self._add_controls()
        self._add_buttons()

    def _add_graph(self):
        box = gui.vBox(self.mainArea)
        self.graph = SlicerPlot(self, self.show_profiles,
                                self.show_range, self.show_mean)
        self.graph.selectionChanged.connect(self.__selection_changed)
        box.layout().addWidget(self.graph)

    def __selection_changed(self, selection: List[Tuple[float, float]]):
        annotated_selection = []
        if selection:
            if self.selection:
                annotated_selection.extend(self.selection)
            last_slice = [list(selection[-1]), f"Slice {len(selection)}"]
            annotated_selection.append(last_slice)
        if self.selection != annotated_selection:
            self.selection = annotated_selection
            self._slice_picker.set_data(self.selection)
            self.commit.deferred()

    def _add_controls(self):
        common_options = dict(labelWidth=50, orientation=Qt.Horizontal,
                              sendSelectedValue=True, contentsLength=12,
                              searchable=True)

        axes_box = gui.vBox(self.controlArea, "Axes")
        gui.comboBox(axes_box, self, "x_var", label="Axis x:",
                     callback=self.__on_x_var_changed,
                     model=self._xy_vars_model, **common_options)
        gui.comboBox(axes_box, self, "y_var", label="Axis y:",
                     callback=self.setup_plot,
                     model=self._xy_vars_model, **common_options)

        key_box = gui.vBox(self.controlArea, "Split by")
        self._key_vars_view = view = ListViewSearch(
            selectionMode=ListViewSearch.ExtendedSelection,
            sizePolicy=QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding),
            minimumSize=QSize(30, 100),
        )
        view.setModel(self._key_vars_model)
        view.selectionModel().selectionChanged.connect(
            self.__on_key_vars_changed
        )
        key_box.layout().addWidget(view)

        groupby_box = gui.vBox(self.controlArea, "Group by")
        gui.comboBox(groupby_box, self, "group_var", callback=self.setup_plot,
                     model=self._group_var_model, **common_options)

        display_box = gui.widgetBox(self.controlArea, "Display")
        gui.checkBox(display_box, self, "show_profiles", "Lines",
                     callback=self.__on_show_profiles_changed,
                     tooltip="Plot lines")
        gui.checkBox(display_box, self, "show_range", "Range",
                     callback=self.__on_show_range_changed,
                     tooltip="Plot range between 10th and 90th percentile")
        gui.checkBox(display_box, self, "show_mean", "Mean",
                     callback=self.__on_show_mean_changed,
                     tooltip="Plot mean curve")

        slices_box = gui.vBox(self.controlArea, "Slices")
        self._slice_picker = SlicePicker(self)
        self._slice_picker.sigSlicesChanged.connect(self.__on_slices_changed)
        self._slice_picker.sigGeometryChanged.connect(
            self.__update_scroll_area_geometry
        )
        slices_box.layout().addWidget(self._slice_picker)

    def __on_x_var_changed(self):
        self.selection = None
        self.setup_plot()
        self.__selection_changed([])

    def __on_key_vars_changed(self):
        rows = self._key_vars_view.selectionModel().selectedRows()
        values = self._key_vars_model[:]
        self.key_vars = [values[row.row()] for row in rows]
        self.setup_plot()

    def __on_show_profiles_changed(self):
        self.check_display_options()
        self.graph.set_show_lines(self.show_profiles)

    def __on_show_range_changed(self):
        self.check_display_options()
        self.graph.set_show_range(self.show_range)

    def __on_show_mean_changed(self):
        self.check_display_options()
        self.graph.set_show_mean(self.show_mean)

    def __on_slices_changed(self, selection: List[Tuple[float, float, str]]):
        if self.selection != selection:
            self.selection = selection
            self.graph.apply_selection([s for s, _ in self.selection])
            self.graph.replot()
            self.commit.deferred()

    def __update_scroll_area_geometry(self):
        for child in self.left_side.children():
            if isinstance(child, QScrollArea):
                child.updateGeometry()
                break

    def _add_buttons(self):
        plot_gui = OWPlotGUI(self)
        plot_gui.box_zoom_select(self.buttonsArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    @Inputs.data
    @check_sql_input
    def set_data(self, data: Optional[Table]):
        self.closeContext()
        self.data = data
        self.clear()
        self.check_data()
        self.check_display_options()
        self.enable_controls()
        self.init_models()
        self.openContext(self.data)
        self.set_list_view_selection()
        self.clip_selection()
        self.setup_plot()
        self.commit.now()

    def clear(self):
        self.key_vars.clear()
        self.graph.clear_plot()
        self._slice_picker.clear_all()
        self._xy_vars_model.set_domain(None)
        self._key_vars_model.set_domain(None)
        self._group_var_model.set_domain(None)

    def check_data(self):
        self.clear_messages()

        if not self.data:
            return

        if not self.data.domain.has_continuous_attributes(True, True):
            self.Warning.no_continuous_vars()
            self.data = None

    def check_display_options(self):
        self.Warning.no_display_option.clear()
        if self.data is not None:
            if not (self.show_profiles or self.show_range or self.show_mean):
                self.Warning.no_display_option()

    def enable_controls(self):
        self._slice_picker.set_add_enabled(bool(self.data))

    def init_models(self):
        domain = self.data.domain if self.data is not None else None

        self._xy_vars_model.set_domain(domain)
        if self._xy_vars_model:
            self.x_var = self._xy_vars_model[0]
            self.y_var = self._xy_vars_model[1] \
                if len(self._xy_vars_model) > 1 else self.x_var

        self._key_vars_model.set_domain(domain)

        self._group_var_model.set_domain(domain)
        if self._group_var_model:
            self.group_var = self._group_var_model[1] \
                if len(self._group_var_model) > 1 else self._group_var_model[0]

    def set_list_view_selection(self):
        selection = QItemSelection()
        sel_model: QItemSelectionModel = self._key_vars_view.selectionModel()
        with disconnected(sel_model.selectionChanged,
                          self.__on_key_vars_changed):
            valid_key_vars = []
            key_vars_model_values = self._key_vars_model[:]
            for var in self.key_vars:
                if var in key_vars_model_values:
                    index = key_vars_model_values.index(var)
                    model_index = self._key_vars_view.model().index(index, 0)
                    selection.append(QItemSelectionRange(model_index))
                    valid_key_vars.append(var)
            self.key_vars = valid_key_vars
            sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

    def clip_selection(self):
        if self.data and self.selection:
            values = self.data.get_column(self.x_var)
            min_val, max_val = np.nanmin(values), np.nanmax(values)
            self.selection = [((max(start, min_val), min(stop, max_val)), name)
                              for (start, stop), name in self.selection
                              if start <= max_val and stop >= min_val]

    def setup_plot(self):
        self.Error.no_valid_data.clear()
        self.Error.non_unique_series.clear()
        self.Information.hidden_instances.clear()
        self.graph.clear_plot()
        self._slice_picker.clear_all()
        if not self.data:
            return
        if self.x_var not in self.data.domain or \
                self.y_var not in self.data.domain or \
                self.group_var and self.group_var not in self.data.domain:
            # TODO - this should not happen
            return

        # TODO - move this also
        x_data = self.data.get_column(self.x_var).astype(float)
        y_data = self.data.get_column(self.y_var).astype(float)
        keys_data = [self.data.get_column(var).astype(float)
                     for var in self.key_vars]

        valid_data_mask = np.isnan(x_data) | np.isnan(y_data)
        for k in keys_data:
            valid_data_mask |= np.isnan(k)

        group_data = None
        if self.group_var:
            group_data = self.data.get_column(self.group_var)
            group_data = group_data.astype(float)
            valid_data_mask |= np.isnan(group_data)

        valid_data_mask = ~valid_data_mask
        if not np.sum(valid_data_mask):
            self.Error.no_valid_data()
            return
        elif not np.all(valid_data_mask):
            self.Information.hidden_instances()

        x_data_valid = x_data[valid_data_mask]
        minimum = np.min(x_data_valid)
        maximum = np.max(x_data_valid)
        self._slice_picker.set_parameters(minimum, maximum, self.x_var)

        self.start(Runner().run,
                   x_data_valid, y_data[valid_data_mask],
                   [k[valid_data_mask] for k in keys_data],
                   None if group_data is None else group_data[valid_data_mask],
                   self.group_var)

    def on_done(self, result: Result):
        if result is None:
            return
        self.graph.set_data(result, self.x_var, self.y_var, self.group_var)
        self.apply_selection()

    def on_partial_result(self, result: Result):
        pass

    def on_exception(self, ex: Exception):
        if isinstance(ex, NonUniqueSeries):
            self.Error.non_unique_series()
        else:
            raise ex

    def apply_selection(self):
        selection = self.__pending_selection or self.selection
        if self.data and selection:
            self.graph.apply_selection([s for s, _ in selection])
            self._slice_picker.set_data(selection)
            self.selection = selection
            self.__pending_selection = None

    @gui.deferred
    def commit(self):
        self.Error.domain_duplicates.clear()
        data = self.data
        if self.data:
            if self.selection:
                slice_var_values = list(n for _, n in self.selection)
                if len(slice_var_values) > len(set(slice_var_values)):
                    self.Error.domain_duplicates()
                    self.Outputs.selected_data.send(None)
                    return
            data = _create_output_table(self.data, self.x_var, self.selection)
            sp = SlicerPreprocessor(self.x_var, self.selection)
            add_transformation_to_data(data, sp, self.data)
        self.Outputs.selected_data.send(data)

    def send_report(self):
        if self.data is None:
            return

        self.report_plot()
        key = None
        if self.key_vars:
            key = ", ".join([v.name for v in self._key_vars_model
                             if v in self.key_vars])
        caption = report.render_items_vert(
            (("X", self.x_var),
             ("Y", self.y_var),
             ("Key", key),
             ("Group by", self.group_var))
        )
        if caption:
            self.report_caption(caption)

    def set_visual_settings(self, key, value):
        self.graph.plot_setter.set_parameter(key, value)
        self.visual_settings[key] = value


class SlicerPreprocessor(Transformation):
    def __init__(self, x_var, selection):
        super().__init__()
        self.x_var = x_var
        self.selection = selection
        # domain after transformation made by widget
        # to know if any Orange transformation happened after
        self.domain = None

    def __call__(self, data: Table) -> Table:
        if self.x_var not in data.domain:
            raise ValueError(
                "The Series Slicer transformation expects data to contain "
                f"{self.x_var.name} variable, which is missing in the data."
            )
        result = _create_output_table(data, self.x_var, self.selection)
        return result

    def __repr__(self):
        if self.selection:
            slc = "<br>".join(f"{n}: {f:.2f} - {t:.2f}" for (f, t), n in self.selection)
        else:
            slc = "N/A"
        table = (("Axis x", self.x_var), ("Slices", slc))
        return f"<h4>Series Slicer</h4>{create_info_html_table(table)}"


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    path = os.path.join(os.path.normpath(path), "datasets", "airpassengers.csv")
    WidgetPreview(OWSlicer).run(set_data=Table(path))
