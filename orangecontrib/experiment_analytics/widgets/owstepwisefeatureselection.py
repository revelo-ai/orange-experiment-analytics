import math
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

import numpy as np
from AnyQt.QtCore import (
    QAbstractTableModel,
    QLocale,
    QModelIndex,
    QObject,
    QRect,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    pyqtSignal,
)
from AnyQt.QtGui import QMouseEvent, QPainter, QFont
from AnyQt.QtWidgets import (
    QHeaderView,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionButton,
    QTableView,
    QWidget,
)

from Orange.base import Learner, Model
from Orange.data import (
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    HasClass,
    Table,
    Variable,
)
from Orange.preprocess import Preprocess
from Orange.widgets import gui
from Orange.widgets.utils import enum2int
from Orange.widgets.utils.concurrent import ConcurrentWidgetMixin, TaskState
from Orange.widgets.widget import OWWidget
from orangewidget.settings import Setting
from orangewidget.utils.itemmodels import PyTableModel
from orangewidget.utils.signals import Input, Output
from orangewidget.widget import Msg

from orangecontrib.experiment_analytics.stepwise_feature_selection import (
    FeatureSelectionPreprocessor,
    Scoring,
    StepwiseFeatureSelection,
    Stopping,
    select_learner,
    supported_types,
)
from orangecontrib.experiment_analytics.transformation_export import (
    ComputeValueTransform,
    add_transformation_to_data,
    create_info_html_table,
)


class ScoresTableModel(PyTableModel):
    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        return super().data(index, role)


class ColoredBarItemDelegate(gui.ColoredBarItemDelegate):
    """
    Use the delegate only for column 4. If ColoredBarItemDelegate's paint used
    on checkbox columns it causes that checkboxes are not shown.
    """

    def paint(self, painter, option, index):
        if index.column() == 3:
            super().paint(painter, option, index)
        else:
            QStyledItemDelegate.paint(self, painter, option, index)

    def displayText(self, value, locale=QLocale()):
        if isinstance(value, float) and not math.isnan(value) and -1e-4 < value < 1e-4:
            return f"{value:.3e}"
        return super().displayText(value, locale)


class HeaderWithCheckbox(QHeaderView):
    """Header that plot checkboxes in the first two columns"""

    # https://forum.qt.io/topic/127103/add-a-qcheckbox-as-a-header-to-a-qtablewidget/21

    CHECK_SYMBOLS = {
        Qt.Checked: QStyle.State_On,
        Qt.Unchecked: QStyle.State_Off,
        Qt.PartiallyChecked: QStyle.State_NoChange,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setSectionsClickable(True)

    def paintSection(self, painter: Optional[QPainter], rect: QRect, index: int):
        painter.save()
        super().paintSection(painter, rect, index)
        painter.restore()
        if index in (0, 1):
            checkbox_rect, option = self.__create_checkbox_rect()
            checkbox_rect.moveLeft(rect.left() + 3)
            option.rect = checkbox_rect

            model = self.model()
            checked = model.headerData(index, self.orientation(), Qt.CheckStateRole)
            option.state = self.CHECK_SYMBOLS[checked]
            self.style().drawPrimitive(QStyle.PE_IndicatorCheckBox, option, painter)

    def mousePressEvent(self, e: QMouseEvent = None):
        super().mousePressEvent(e)
        section = self.logicalIndexAt(e.pos())
        if section in (0, 1) and self.orientation() == Qt.Horizontal:
            checkbox_rect, _ = self.__create_checkbox_rect()
            checkbox_rect.moveLeft(self.sectionViewportPosition(section) + 3)
            if checkbox_rect.contains(e.pos()):
                self.model().sourceModel().set_next_state(section)
                self.viewport().update()

    def __create_checkbox_rect(self):
        option = QStyleOptionButton()
        option.initFrom(self)
        cb_indicator = QStyle.SubElement.SE_CheckBoxIndicator
        return self.style().subElementRect(cb_indicator, option, self), option


class FeatureTable(QTableView):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, sortingEnabled=True)
        self.setItemDelegate(ColoredBarItemDelegate(self, decimals=4))
        self.setHorizontalHeader(HeaderWithCheckbox(Qt.Horizontal, self))
        self.setMinimumSize(500, 200)

    def setModel(self, model: Optional[QAbstractTableModel]):
        super().setModel(model)
        stc = QHeaderView.ResizeToContents
        for col, mode in enumerate((stc, stc, QHeaderView.Stretch, stc)):
            self.horizontalHeader().setSectionResizeMode(col, mode)


class FeatureTableModel(QAbstractTableModel):
    LOCKED_COL, ENTERED_COL, FEATURES_COL, SCORE_COL = range(4)
    COLUMNS = ["    Locked", "    Entered", "Feature", "Score difference"]

    variable_locked = pyqtSignal(set)
    variable_unlocked = pyqtSignal(set)
    variable_entered = pyqtSignal(set)
    variable_excluded = pyqtSignal(set)

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        self.attributes = ()
        self.scores = {}
        self.locked = set()
        self.entered = set()
        self.max_row = None

    def columnCount(self, parent: QModelIndex = ..., *args, **kwargs) -> int:
        return 4

    def rowCount(self, parent: QModelIndex = ..., *args, **kwargs) -> int:
        return len(self.attributes)

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        col, row = index.column(), index.row()
        # non cell should be selectable
        flags = super().flags(index) & ~Qt.ItemIsSelectable
        if col == 0 or (col == 1 and self.attributes[row] not in self.locked):
            # first column always checkable, second if attribute is not locked
            flags |= Qt.ItemIsUserCheckable
        return flags

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.COLUMNS[section]
        if orientation == Qt.Horizontal and role == Qt.CheckStateRole:
            set_ = [self.locked, self.entered][section]
            if not set_:
                return Qt.Unchecked
            elif len(set(self.attributes) - set_) > 0:
                return Qt.PartiallyChecked
            else:
                return Qt.Checked
        return super().headerData(section, orientation, role)

    def set_next_state(self, section: int):
        """Function returns what should be the next state of header checkbox"""
        all_attrs = set(self.attributes)
        if section == self.LOCKED_COL:
            uncheck = all_attrs == self.locked
        elif section == self.ENTERED_COL:
            uncheck = (all_attrs - self.locked) <= self.entered
        new_state = Qt.Unchecked if uncheck else Qt.Checked
        self.setHeaderData(section, Qt.Horizontal, new_state, Qt.CheckStateRole)

    def setHeaderData(
        self,
        section: int,
        orientation: Qt.Orientation,
        value: Any,
        role: int = Qt.DisplayRole,
    ):
        if role == Qt.CheckStateRole:
            if section == self.ENTERED_COL:
                if value == Qt.Checked:
                    # add all not entered except locked
                    to_add = set(self.attributes) - self.entered - self.locked
                    self.variable_entered.emit(to_add)
                    self.entered |= to_add
                else:
                    # remove all entered while keeping entered and locked
                    to_remove = self.entered - (self.entered & self.locked)
                    self.variable_excluded.emit(to_remove)
                    self.entered -= to_remove
            elif section == self.LOCKED_COL:
                new_val = set(self.attributes) if value == Qt.Checked else set()
                if value == Qt.Checked:
                    self.variable_locked.emit(new_val - self.locked)
                else:
                    self.variable_unlocked.emit(self.locked)
                self.locked = new_val

            index_top = self.index(0, section)
            index_bottom = self.index(len(self.attributes) - 1, section)
            self.dataChanged.emit(index_top, index_bottom, (Qt.CheckStateRole,))
            return True
        return super().setHeaderData(section, orientation, value, role)

    def data(self, index, role: int = Qt.DisplayRole) -> Any:
        if index.isValid():
            col = index.column()
            row = index.row()
            attr = self.attributes[row]

            if role == gui.BarRatioRole and col == self.SCORE_COL:
                # value bar in score column
                value = super().data(index, Qt.DisplayRole)
                if value == "":
                    return None
                vmin, vmax = min(self.scores.values()), max(self.scores.values())
                # when only one bar in column make it 1 unit long
                return ((value - vmin) / (vmax - vmin)) if vmax - vmin > 0 else 1

            if role == Qt.CheckStateRole:
                # checkbox columns
                if col == self.LOCKED_COL:
                    return Qt.Checked if attr in self.locked else Qt.Unchecked
                elif col == self.ENTERED_COL:
                    return Qt.Checked if attr in self.entered else Qt.Unchecked
            elif role == Qt.DisplayRole:
                # columns with features and scores
                if col == self.FEATURES_COL:
                    return attr.name
                elif col == self.SCORE_COL:
                    return self.scores.get(attr, "")
            elif role == Qt.FontRole and len(self.scores) > 0 and row == self.max_row:
                # set bold font for the line with the highest score
                font = QFont()
                font.setBold(True)
                return font

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.EditRole) -> bool:
        if role == Qt.CheckStateRole:
            col, row = index.column(), index.row()
            var = self.attributes[row]
            if col == self.ENTERED_COL:
                if value:
                    self.entered.add(var)
                    self.variable_entered.emit({var})
                else:
                    self.entered.remove(var)
                    self.variable_excluded.emit({var})
            if col == self.LOCKED_COL:
                if value:
                    self.locked.add(var)
                    self.variable_locked.emit({var})
                else:
                    self.locked.remove(var)
                    self.variable_unlocked.emit({var})
            self.headerDataChanged.emit(Qt.Horizontal, col, col)
            return True
        else:
            return super().setData(index, value, role)

    def setup_table(self, data, locked, entered):
        self.beginResetModel()
        self.attributes = data.domain.attributes
        self.locked = locked.copy()
        self.entered = entered.copy()
        self.endResetModel()

    def update_entered(self, entered: Set[Variable]):
        # get added and removed features
        changed_features = (entered - self.entered) | (self.entered - entered)
        self.entered = entered.copy()
        for f in changed_features:
            index = self.index(self.attributes.index(f), self.ENTERED_COL)
            self.dataChanged.emit(index, index, (Qt.CheckStateRole,))

    def update_locked(self, locked: Set[Variable]):
        # get added and removed features
        changed_features = (locked - self.locked) | (self.locked - locked)
        self.locked = locked.copy()
        for f in changed_features:
            index = self.index(self.attributes.index(f), self.LOCKED_COL)
            self.dataChanged.emit(index, index, (Qt.CheckStateRole,))

    def update_scores(self, scores: Dict[Variable, float]):
        self.scores = scores.copy()
        # find the max score row bolding
        self.max_row = None
        scores = [self.scores.get(a, -float("inf")) for a in self.attributes]
        if len(scores) and np.any(~np.isinf(scores)):
            self.max_row = np.argmax(scores)

        # emmit scores changed - for bars and scores
        idx_top = self.index(0, self.SCORE_COL)
        idx_bottom = self.index(len(self.attributes) - 1, self.SCORE_COL)
        self.dataChanged.emit(idx_top, idx_bottom, (gui.BarRatioRole, Qt.DisplayRole))
        # emmit change for bolding (FontRole)
        idx_bottom_right = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(self.index(0, 0), idx_bottom_right, (Qt.FontRole,))

    def clear(self):
        self.beginResetModel()
        self.attributes = ()
        self.scores = {}
        self.locked = set()
        self.entered = set()
        self.endResetModel()


class FeaturesProxyModel(QSortFilterProxyModel):
    """Model that sort last columns so that empty values always at bottom"""

    def lessThan(self, left_ind: QModelIndex, right_ind: QModelIndex) -> bool:
        if left_ind.column() == 3 and right_ind.column() == 3:
            m = self.sourceModel()
            left = m.data(m.index(left_ind.row(), 1), role=Qt.CheckStateRole)
            right = m.data(m.index(right_ind.row(), 1), role=Qt.CheckStateRole)
            if left != right:
                # when one checked and other not - sort checked always top
                is_ascending = self.sortOrder() == Qt.AscendingOrder
                return (not is_ascending) if right == Qt.Checked else is_ascending
        if left_ind.column() in (0, 1) and right_ind.column() in (0, 1):
            # sort first and second column by check status
            left = self.sourceModel().data(left_ind, role=Qt.CheckStateRole)
            right = self.sourceModel().data(right_ind, role=Qt.CheckStateRole)
            return (left == Qt.Checked) < (right == Qt.Checked)
        return super().lessThan(left_ind, right_ind)


# todo: remove following two classes in the future when TaskState implements
#  backward progress functionality
class StepwiseTaskState(TaskState):
    """TaskState that also go back in progres to run multiple times between 0 and 100"""

    def __init__(self, *args):
        super().__init__(*args)
        self.__progress = 0

    def set_progress_value(self, value: float):
        if round(value, 1) != round(self.__progress, 1):
            # Only emit progress when it has changed sufficiently
            self._p_progress_changed.emit(value)
            self.__progress = value


class StepwiseConcurrentWidgetMixin(ConcurrentWidgetMixin):
    """Make ConcurrentWidgetMixin use slightly modified TaskSate"""

    def start(self, task: Callable, *args, **kwargs):
        self._ConcurrentWidgetMixin__set_state_ready()
        self._ConcurrentMixin__cancel_task(wait=False)
        assert callable(task), "`task` must be callable!"
        state = StepwiseTaskState(self)
        task = partial(task, *(args + (state,)), **kwargs)
        self._ConcurrentMixin__start_task(task, state)
        self._ConcurrentWidgetMixin__set_state_busy()


class OWStepwiseFeatureSelection(OWWidget, StepwiseConcurrentWidgetMixin):
    name = "Stepwise Feature Selection"
    icon = "icons/stepwiseFeatureSelection.svg"
    priority = 120

    class Inputs:
        data = Input("Data", Table)
        learner = Input("Learner", Learner)

    class Outputs:
        data = Output("Data", Table)
        preprocessor = Output("Preprocessor", Preprocess)
        model = Output("Model", Model)

    class Error(OWWidget.Error):
        class_required = Msg("Data input requires a target variable.")
        unexpected_error = Msg("{}")

    class Warning(OWWidget.Warning):
        nan_class = Msg("Instances with unknown target values were removed from data.")
        nan_col = Msg("Features with all unknown values were removed from data.")

    DEFAULT_SORTING = (-1, enum2int(Qt.AscendingOrder))

    validation_method: str = Setting(next(iter(Scoring.VALIDATION_METHODS)))
    cv_num_folds: int = Setting(5)
    rs_test_size: int = Setting(10)
    direction: str = Setting("Forward")
    stopping_rule: str = Setting(next(iter(Stopping.RULES)))
    st_rule_num_features: int = Setting(5)
    st_rule_threshold: float = Setting(0.1)
    sort_column_order: Tuple[int, int] = Setting(DEFAULT_SORTING)
    scoring_method: str = Setting(next(iter(Scoring.ALL_SCORING_METHODS)))
    auto_commit: bool = Setting(True)

    entered_features: List[Variable] = Setting([], schema_only=True)
    locked_features: List[Variable] = Setting([], schema_only=True)

    def __init__(self):
        super().__init__()
        ConcurrentWidgetMixin.__init__(self)
        self.learner: Optional[Learner] = None
        self.data: Optional[Table] = None
        self.stepwise_fs = StepwiseFeatureSelection(
            self.direction, self.scoring_method, self.__get_validation()
        )

        self.num_entered_attr = 0
        self.num_all_attr = 0

        self.__pending_entered: List[Variable] = self.entered_features
        self.__pending_locked: List[Variable] = self.locked_features

        self.__setup_control_area()
        self.__setup_main_area()
        # call after gui rendered otherwise label shows out of the box
        QTimer.singleShot(1, self.__update_validation_gui)
        QTimer.singleShot(1, self.__update_stopping_gui)
        QTimer.singleShot(1, self.__init_score_combo)
        self.commit.now()

    def __setup_control_area(self):
        vbox = gui.vBox(self.controlArea, "Internal validation")
        gui.comboBox(
            vbox,
            self,
            "validation_method",
            items=list(Scoring.VALIDATION_METHODS),
            sendSelectedValue=True,
            callback=self.__validation_changed,
            label="Method",
        )

        hbox1 = gui.hBox(None)
        gui.label(hbox1, self, "Folds:")
        gui.spin(
            hbox1,
            self,
            "cv_num_folds",
            minv=2,
            maxv=10,
            callback=self.__validation_changed,
        )
        hbox2 = gui.hBox(None)
        gui.label(hbox2, self, "Test set size:")
        gui.spin(
            hbox2,
            self,
            "rs_test_size",
            minv=1,
            maxv=99,
            step=5,
            callback=self.__validation_changed,
            suffix="  %",
        )

        hbox = gui.indentedBox(vbox, orientation=Qt.Horizontal)
        self.val_stacked_widget = QStackedWidget()
        hbox.layout().addWidget(self.val_stacked_widget)
        self.val_stacked_widget.addWidget(hbox1)
        self.val_stacked_widget.addWidget(hbox2)

        vbox = gui.vBox(self.controlArea, "Feature selection")
        self.scoring_method_cb = gui.comboBox(
            vbox,
            self,
            "scoring_method",
            label="Score",
            items=[],
            sendSelectedValue=True,
            callback=self.__score_changed,
        )

        gui.comboBox(
            vbox,
            self,
            "direction",
            label="Direction",
            items=StepwiseFeatureSelection.DIRECTIONS,
            sendSelectedValue=True,
            callback=self.__direction_changed,
        )
        gui.comboBox(
            vbox,
            self,
            "stopping_rule",
            label="Stopping rule",
            items=list(Stopping.RULES),
            sendSelectedValue=True,
            callback=[self.__update_stopping_gui, self.commit.deferred],
        )

        hbox1 = gui.hBox(None)
        gui.label(hbox1, self, "Number features:")
        self.num_features_spin = gui.spin(
            hbox1,
            self,
            "st_rule_num_features",
            minv=1,
            maxv=1000,
            callback=self.commit.deferred,
        )
        hbox2 = gui.hBox(None)
        gui.label(hbox2, self, "Threshold:")
        gui.doubleSpin(
            hbox2,
            self,
            "st_rule_threshold",
            minv=0,
            maxv=1000,
            step=0.1,
            decimals=4,
            callback=self.commit.deferred,
        )

        hbox = gui.indentedBox(vbox, orientation=Qt.Horizontal)
        self.stopping_stacked_widget = QStackedWidget()
        hbox.layout().addWidget(self.stopping_stacked_widget)
        self.stopping_stacked_widget.addWidget(hbox1)
        self.stopping_stacked_widget.addWidget(hbox2)
        self.stopping_stacked_widget.addWidget(QWidget())
        self.stopping_stacked_widget.addWidget(QWidget())

        self.start_btn = gui.button(
            self.controlArea, self, "Start", callback=self.__start
        )
        hbox = gui.hBox(self.controlArea)
        self.undo_btn = gui.button(hbox, self, "Undo", callback=self.__undo)
        self.step_btn = gui.button(hbox, self, "Step", callback=self.__step)

        gui.rubber(self.controlArea)
        gui.auto_send(self.buttonsArea, self, "auto_commit")

    def __setup_main_area(self):
        vbox = gui.vBox(self.mainArea, "Model statistics")
        gui.label(
            vbox,
            self,
            "Number entered attributes: %(num_entered_attr)i/%(num_all_attr)i",
        )
        vbox.setFixedHeight(120)
        self.scores_model = ScoresTableModel()
        self.scores_table = QTableView()
        self.scores_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.scores_table.verticalHeader().hide()
        self.scores_table.setModel(self.scores_model)
        vbox.layout().addWidget(self.scores_table)

        vbox = gui.vBox(self.mainArea, "Current estimates")
        self.features_table = FeatureTable()
        self.features_model = FeatureTableModel()
        self.features_model.variable_entered.connect(self.__variable_entered)
        self.features_model.variable_excluded.connect(self.__variable_excluded)
        self.features_model.variable_locked.connect(self.__variable_locked)
        self.features_model.variable_unlocked.connect(self.__variable_unlocked)
        proxy_model = FeaturesProxyModel()
        proxy_model.setSourceModel(self.features_model)
        self.features_table.setModel(proxy_model)
        header = self.features_table.horizontalHeader()
        header.sectionClicked.connect(self.__horizontal_header_clicked)
        # PyQt6's SortOrder is Enum (and not IntEnum as in PyQt5),
        # transform sort_column_order[1], which is int, in Qt.SortOrder Enum
        sco = (self.sort_column_order[0], Qt.SortOrder(self.sort_column_order[1]))
        header.setSortIndicator(*sco)
        vbox.layout().addWidget(self.features_table)

    def __score_changed(self):
        self.start(self.__run, partial(self.stepwise_fs.set_score, self.scoring_method))

    def __validation_changed(self):
        self.__update_validation_gui()
        val = self.__get_validation()
        self.start(self.__run, partial(self.stepwise_fs.set_validation, val))

    def __direction_changed(self):
        self.start(self.__run, partial(self.stepwise_fs.set_direction, self.direction))

    def __start(self):
        if self.task is None:
            self.start_btn.setText("Stop")
            self.start(self.stepwise_fs.run, self.__get_stopping())
        else:
            self.stepwise_fs.stop_run()
            # change button back to start and lock it until distances full recompute
            self.start_btn.setText("Start")
            self.__set_buttons_disabled(True)

    def __step(self):
        self.start(self.stepwise_fs.step)

    def __undo(self):
        self.start(self.__run, self.stepwise_fs.step_back)

    def __variable_entered(self, variables: Set[Variable]):
        self.entered_features += list(variables)
        self.start(self.stepwise_fs.include, variables)

    def __variable_excluded(self, variables: Set[Variable]):
        self.entered_features = list(set(self.entered_features) - variables)
        self.start(self.stepwise_fs.exclude, variables)

    def __variable_locked(self, variables: Set[Variable]):
        self.locked_features += list(variables)
        self.stepwise_fs.lock(variables)
        self.features_model.update_scores(self.stepwise_fs.scores)

    def __variable_unlocked(self, variables: Set[Variable]):
        self.locked_features = list(set(self.locked_features) - variables)
        self.stepwise_fs.unlock(variables)
        self.features_model.update_scores(self.stepwise_fs.scores)

    def __horizontal_header_clicked(self, index: int):
        header = self.features_table.horizontalHeader()
        self.sort_column_order = (index, enum2int(header.sortIndicatorOrder()))

    @Inputs.data
    def set_data(self, data: Table):
        if self.data is not None:
            self.__pending_entered = self.entered_features
            self.__pending_locked = self.locked_features
        self.Error.clear()
        self.Warning.clear()

        self.__check_data(data)
        self.__init_score_combo()
        self.__clear()

        if self.data is not None:
            if len(self.__pending_entered) > 0:
                self.entered_features = [v for v in self.__pending_entered
                                         if v in self.data.domain.attributes]
            if len(self.__pending_locked) > 0:
                self.locked_features = [v for v in self.__pending_locked
                                        if v in self.data.domain.attributes]

            self.features_model.setup_table(
                self.data, set(self.locked_features), set(self.entered_features)
            )
            self.__update_stopping_gui()
            self.num_all_attr = len(self.features_model.attributes)
            self.num_entered_attr = len(self.entered_features)
        self.start(
            self.__run,
            partial(
                self.stepwise_fs.set_data_and_scoring,
                self.data,
                set(self.entered_features),
                set(self.locked_features),
                self.scoring_method,
            ),
        )
        self.commit.now()

    def __clear(self):
        self.cancel()
        self.entered_features = []
        self.locked_features = []
        self.features_model.clear()
        self.scores_model.clear()
        self.num_entered_attr = 0
        self.num_all_attr = 0

    def __check_data(self, data: Table):
        if data and not data.domain.class_var:
            self.Error.class_required()
            self.data = None
        else:
            self.data = data

        if self.data is not None:
            # remove rows with nan in a class variable
            if np.isnan(data.Y).any():
                self.Warning.nan_class()
                self.data = HasClass()(self.data)

            # remove only nan columns
            all_nan = np.all(np.isnan(self.data), axis=0)
            if len(self.data) and np.any(all_nan):
                self.Warning.nan_col()
                domain = self.data.domain
                att = [a for a, n in zip(domain.attributes, all_nan) if not n]
                new_domain = Domain(att, domain.class_vars, domain.metas)
                self.data = self.data.transform(new_domain)

    @Inputs.learner
    def set_learner(self, learner: Learner):
        self.learner = learner
        self.__init_score_combo()
        self.start(self.__run, partial(self.stepwise_fs.set_learner, learner))

    def start(self, run_fun: Callable, *args):
        self.Error.unexpected_error.clear()
        # when computation is happening lock all controls to wait until scores are
        # computed. When start button clicked do not lock it to enable stopping
        self.__set_buttons_disabled(True, run_fun != self.stepwise_fs.run)
        super().start(run_fun, *args)

    @staticmethod
    def __run(fun, task_state: TaskState):
        def callback(progress: float):
            task_state.set_progress_value(progress * 100)

        fun(callback)

    def on_done(self, _: Any = None):
        self.replot()
        self.start_btn.setText("Start")
        self.commit.deferred()
        self.__set_buttons_disabled(False)

    def on_partial_result(self, scope: str):
        if scope == "entered":
            self.replot_entered()
        else:
            self.replot()

    def on_exception(self, ex: Exception):
        self.Error.unexpected_error(str(ex))
        self.start_btn.setText("Start")
        self.commit.deferred()
        # something is wrong with either data or learner, disable stopping
        self.__set_buttons_disabled(True)

    def replot(self):
        self.replot_entered()
        self.replot_locked()
        self.replot_scores()

    def replot_entered(self):
        self.entered_features = list(self.stepwise_fs.selected)
        self.features_model.update_entered(set(self.entered_features))

    def replot_locked(self):
        self.locked_features = list(self.stepwise_fs.locked)
        self.features_model.update_locked(set(self.locked_features))

    def replot_scores(self):
        self.features_model.update_scores(self.stepwise_fs.scores)
        scores = self.stepwise_fs.compute_scores()
        self.scores_model.setHorizontalHeaderLabels(list(scores))
        self.scores_model.wrap([list(scores.values())])
        self.num_entered_attr = len(self.entered_features)

    @gui.deferred
    def commit(self):
        data = None
        model = None
        if self.data:
            domain = Domain(
                [a for a in self.data.domain.attributes if a in self.entered_features],
                self.data.domain.class_vars,
                self.data.domain.metas,
            )
            data = self.data.transform(domain)
            add_transformation_to_data(
                data, StepwiseFeatureSelectionTransform(data.domain), self.data
            )
            learner = self.stepwise_fs.scorer.get_learner(data)
            model = select_learner(data, learner)(data)
        self.Outputs.data.send(data)
        self.Outputs.model.send(model)
        preprocessor = FeatureSelectionPreprocessor(
            direction=self.direction,
            scoring_method=self.scoring_method,
            validation=self.__get_validation(),
            stopping_rule=self.__get_stopping(),
            learner=self.learner,
        )
        self.Outputs.preprocessor.send(preprocessor)

    def __get_validation(self) -> Tuple[str, Dict[str, int]]:
        if self.validation_method == "Cross validation":
            params = {"k": self.cv_num_folds}
        else:
            params = {"test_size": self.rs_test_size / 100}
        return self.validation_method, params

    def __get_stopping(self) -> Tuple[str, Dict[str, int]]:
        kwargs = {}
        if self.stopping_rule == "N-features":
            kwargs = {"n_features": self.st_rule_num_features}
        elif self.stopping_rule == "Score delta":
            kwargs = {"threshold": self.st_rule_threshold}
        return self.stopping_rule, kwargs

    def __init_score_combo(self):
        self.scoring_method_cb.clear()
        # since some models can perform both classification and regression set
        # types on based on data class when data available and on model otherwise
        if self.data:
            types = (type(self.data.domain.class_var),)
        elif self.learner:
            types = supported_types(self.learner)
        else:  # when no learner combo should show all methods
            types = (ContinuousVariable, DiscreteVariable)
        methods = Scoring.get_methods(types)
        for it in Scoring.get_methods(types):
            self.scoring_method_cb.addItem(it)
        if self.scoring_method not in methods:
            self.scoring_method = methods[0]

    def __update_validation_gui(self):
        is_cv = self.validation_method == "Random split"
        self.val_stacked_widget.setCurrentIndex(int(is_cv))

    def __update_stopping_gui(self):
        # set maximum for number of features spin to number of features in domain
        n_features = 1000 if self.data is None else len(self.data.domain.attributes)
        self.num_features_spin.setRange(1, n_features)

        # show spin for selected rule
        idx = self.controls.stopping_rule.currentIndex()
        self.stopping_stacked_widget.setCurrentIndex(idx)

    def __set_buttons_disabled(self, is_disabled: bool, disable_start=True):
        if disable_start:
            self.start_btn.setDisabled(is_disabled)
        self.step_btn.setDisabled(is_disabled)
        self.undo_btn.setDisabled(is_disabled)

    def send_report(self):
        self.report_settings()
        if self.data is not None:
            self.report_tables()

    def report_settings(self):
        if self.validation_method == "Cross validation":
            val_param, val_val = "&nbsp;&nbsp;Number folds", self.cv_num_folds
        else:
            val_param, val_val = "&nbsp;&nbsp;Test set size", self.rs_test_size
        st_rule_suf = ""
        if self.stopping_rule == "N-features":
            st_rule_suf = f": {self.st_rule_num_features}"
        elif self.stopping_rule == "Score delta":
            st_rule_suf = f": {self.st_rule_threshold}"
        settings = {
            "Validation method": self.validation_method,
            val_param: val_val,
            "Score": self.scoring_method,
            "Direction": self.direction,
            "Stopping rule": self.stopping_rule + st_rule_suf,
        }

        self.report_items("Settings", settings)

    def report_tables(self):
        self.report_table("Model statistics", self.scores_table)
        fm = self.features_table.model()

        def get_value(row, col):
            role = Qt.DisplayRole if col >= 2 else Qt.CheckStateRole
            data_ = fm.data(fm.index(row, col), role)
            if col < 2:
                data_ = "&check;" if data_ == Qt.Checked else "&cross;"
            return data_

        cols, rows = fm.columnCount(), fm.rowCount()
        data = [[fm.headerData(i, Qt.Horizontal).strip() for i in range(cols)]]
        data += [[get_value(r, c) for c in range(cols)] for r in range(rows)]
        self.report_table("Current estimates", data, header_rows=1)


class StepwiseFeatureSelectionTransform(ComputeValueTransform):
    """
    ComputeValueTransform with own description. This transformation could be
    skipped, since it would be handled through standard ComputeValueTransform
    as it is handled for other Orange widgets. It is implemented only because
    of description in the Save Transformations widget.
    """

    def __repr__(self):
        entered_features = ", ".join(attr.name for attr in self.domain.attributes)
        if not entered_features:
            entered_features = "(none)"
        table = (("Entered features", entered_features),)
        return f"<h4>Stepwise Feature Selection</h4>{create_info_html_table(table)}"


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    housing = Table("housing")
    iris = Table("iris")
    WidgetPreview(OWStepwiseFeatureSelection).run(set_data=iris)
