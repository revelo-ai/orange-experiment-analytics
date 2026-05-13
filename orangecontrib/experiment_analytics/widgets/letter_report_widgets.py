"""
Implements table view to be used in Letter Report widget.
"""

# pylint: disable=missing-docstring,invalid-name,no-name-in-module
from typing import Union

import numpy as np
from AnyQt.QtCore import QAbstractTableModel, Qt, QModelIndex, QPoint
from AnyQt.QtGui import QResizeEvent, QPainter, QPen, QColor
from AnyQt.QtWidgets import QTableView, QScrollBar, QHeaderView, \
    QAbstractItemView, QProxyStyle, QStyleOption, QStyledItemDelegate, \
    QStyle, QStyleOptionViewItem

from Orange.widgets import gui

HeaderRole = next(gui.OrangeUserRole)


class HeaderItemDelegate(QStyledItemDelegate):
    def paint(
            self,
            painter: QPainter,
            option: QStyleOptionViewItem,
            index: QModelIndex
    ):
        QStyledItemDelegate.paint(self, painter, option, index)
        if index.data(HeaderRole):
            painter.save()
            painter.setPen(QPen(QColor(Qt.darkGray), 2))
            rect = option.rect
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())
            painter.restore()


class FrozenHeaderTableView(QTableView):
    """
    Implements a table with its first rows frozen.

    Taken from:
    https://doc.qt.io/qt-5/qtwidgets-itemviews-frozencolumn-example.html
    """
    STYLE_SHEET = "QTableView { border: none;" \
                  "selection-background-color: #999}"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__frozen_header = QTableView(self, *args, **kwargs)
        self.__n_header_rows = 1
        self.__init_frozen_header()

        self.setHorizontalScrollMode(self.ScrollPerPixel)
        self.setVerticalScrollMode(self.ScrollPerPixel)

        header: QHeaderView = self.horizontalHeader()
        header.sectionResized.connect(self.__update_header_width)

        header: QHeaderView = self.__frozen_header.verticalHeader()
        header.sectionResized.connect(self.__update_header_height)

        hscroll: QScrollBar = self.__frozen_header.horizontalScrollBar()
        hscroll.valueChanged.connect(self.horizontalScrollBar().setValue)
        self.horizontalScrollBar().valueChanged.connect(hscroll.setValue)

    def __init_frozen_header(self):
        self.viewport().stackUnder(self.__frozen_header)
        self.__frozen_header.setFocusPolicy(Qt.NoFocus)
        self.__frozen_header.horizontalHeader().hide()
        self.__frozen_header.setHorizontalScrollBarPolicy(
            Qt.ScrollBarAlwaysOff)
        self.__frozen_header.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.__frozen_header.setHorizontalScrollMode(self.ScrollPerPixel)
        self.__frozen_header.show()
        self.__frozen_header.setStyleSheet(self.STYLE_SHEET)
        self.__frozen_header.setItemDelegate(HeaderItemDelegate())

    @property
    def n_header_rows(self) -> int:
        return self.__n_header_rows

    @n_header_rows.setter
    def n_header_rows(self, n_rows: int):
        self.__n_header_rows = n_rows
        for column in range(self.model().columnCount()):
            index = self.model().index(n_rows - 1, column)
            self.model().setData(index, True, HeaderRole)
        width = self.verticalHeader().width()
        self.__frozen_header.verticalHeader().setFixedWidth(width)
        self.__update_frozen_header_rows()
        self.__update_frozen_header_geometry()

    def setModel(self, model: QAbstractTableModel):
        super().setModel(model)
        self.__frozen_header.setModel(model)
        self.__frozen_header.setSelectionModel(self.selectionModel())

    def set_model(self, model: QAbstractTableModel, n_header_rows: int):
        self.setModel(model)
        self.n_header_rows = n_header_rows

    def set_header_spans(self, span_array: np.ndarray, offset=0):
        self.__frozen_header.clearSpans()
        n_rows, n_cols = span_array.shape
        for row in range(n_rows - 1):
            indices = np.flatnonzero(np.diff(span_array[row])) + 1
            indices = [0] + list(indices) + [n_cols]
            for i, next in zip(indices, indices[1:]):
                if next - i > 1:
                    self.__frozen_header.setSpan(row, i + offset, 1, next - i)
        self.__update_frozen_header_geometry()

    def __update_frozen_header_rows(self):
        if not self.model():
            return

        for i in range(self.model().rowCount()):
            self.__frozen_header.setRowHidden(i, i >= self.__n_header_rows)
        for i in range(self.__n_header_rows):
            self.__frozen_header.setRowHeight(i, self.rowHeight(i))

    def __update_frozen_header_geometry(self):
        self.__frozen_header.setGeometry(
            self.frameWidth(),
            self.horizontalHeader().height() + self.frameWidth(),
            self.viewport().width() + self.verticalHeader().width(),
            sum([self.rowHeight(i) for i in range(self.__n_header_rows)])
        )

    def __update_header_width(self, column: int, _: int, width: int):
        self.__frozen_header.setColumnWidth(column, width)
        self.resizeRowsToContents()

    def __update_header_height(self, row: int, _: int, height: int):
        if row < self.__n_header_rows:
            self.setRowHeight(row, height)
            self.__update_frozen_header_geometry()

    def resizeRowsToContents(self):
        super().resizeRowsToContents()
        self.__frozen_header.resizeRowsToContents()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.__update_frozen_header_geometry()
        self.resizeRowsToContents()

    def moveCursor(self, cursor_action: QAbstractItemView.CursorAction,
                   modifiers: Union[Qt.KeyboardModifiers, Qt.KeyboardModifier]
                   ) -> QModelIndex:
        """
        When navigating around the table with the keyboard, ensure that the
        current selection does not disappear behind the frozen rows.
        """
        current: QModelIndex = super().moveCursor(cursor_action, modifiers)
        header_height = sum(self.rowHeight(i)
                            for i in range(self.__n_header_rows))
        if cursor_action == self.MoveUp and \
                current.row() >= self.__n_header_rows and \
                self.visualRect(current).topLeft().y() < header_height:
            value = self.verticalScrollBar().value() + \
                    self.visualRect(current).topLeft().y() - \
                    sum(self.__frozen_header.rowHeight(i)
                        for i in range(self.__n_header_rows))
            self.verticalScrollBar().setValue(value)
        return current


class ScrollableColumnTableView(QTableView):
    """
    Implements a table with its first column scrollable.
    """
    STYLE_SHEET = "QTableView { border: none;" \
                  "selection-color: #FFF;" \
                  "selection-background-color: #0063e1}"

    class DropIndicatorStyle(QProxyStyle):
        def drawPrimitive(
                self,
                element: QStyle.PrimitiveElement,
                option: QStyleOption,
                painter: QPainter,
                widget: QTableView = None
        ):

            if element == self.PE_IndicatorItemViewItemDrop:
                return  # TODO - extend line (or remove it)
                if not option.rect.isNull():
                    painter.drawLine(
                        option.rect.topLeft(),
                        QPoint(widget.width(), option.rect.topLeft().y())
                    )
            else:
                super().drawPrimitive(element, option, painter, widget)

    class ScrollView(QTableView):
        def startDrag(self, actions: Qt.DropAction):
            self.parent().on_drag_start(actions)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__scroll_view = self.ScrollView(self, *args, **kwargs)
        self.__init_scroll_view()

        header: QHeaderView = self.horizontalHeader()
        header.sectionResized.connect(self.__update_scroll_view_width)

        header: QHeaderView = self.__scroll_view.verticalHeader()
        header.sectionResized.connect(self.__update_scroll_view_height)

        vscroll: QScrollBar = self.__scroll_view.verticalScrollBar()
        vscroll.valueChanged.connect(self.__on_view_vscroll_value_changed)
        self.verticalScrollBar().valueChanged.connect(vscroll.setValue)

    def __init_scroll_view(self):
        self.viewport().stackUnder(self.__scroll_view)

        style = self.DropIndicatorStyle()
        self.setStyle(style)
        self.__scroll_view.setStyle(style)

        self.verticalHeader().setDefaultSectionSize(20)
        self.verticalHeader().hide()
        self.__scroll_view.verticalHeader().setDefaultSectionSize(20)
        self.__scroll_view.verticalHeader().hide()

        self.horizontalHeader().hide()
        self.__scroll_view.horizontalHeader().hide()

        self.__scroll_view.setFocusPolicy(Qt.NoFocus)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.__scroll_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setHorizontalScrollMode(self.ScrollPerPixel)
        self.__scroll_view.setHorizontalScrollMode(self.ScrollPerPixel)
        self.setVerticalScrollMode(self.ScrollPerPixel)
        self.__scroll_view.setVerticalScrollMode(self.ScrollPerPixel)

        self.__scroll_view.setStyleSheet(self.STYLE_SHEET)
        self.setStyleSheet(self.STYLE_SHEET)
        self.__scroll_view.show()

    def __on_view_vscroll_value_changed(self, value: int):
        maximum = self.verticalScrollBar().maximum()
        if value <= maximum:
            self.verticalScrollBar().setValue(value)
        else:
            self.__scroll_view.verticalScrollBar().setValue(maximum)

    def setModel(self, model: QAbstractTableModel):
        super().setModel(model)
        if model.columnCount() > 0:
            self.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

        self.__scroll_view.setModel(model)
        self.__scroll_view.setSelectionModel(self.selectionModel())

    def __update_scroll_view_columns(self):
        if self.model().columnCount() == 0:
            return

        header = self.__scroll_view.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        if self.columnWidth(0) > self.__scroll_view.columnWidth(0):
            header.setSectionResizeMode(0, QHeaderView.Fixed)
            self.__scroll_view.setColumnWidth(0, self.columnWidth(0))

        for i in range(self.model().columnCount()):
            self.__scroll_view.setColumnHidden(i, i > 0)

    def __update_scroll_view_geometry(self):
        self.__scroll_view.setGeometry(
            self.verticalHeader().width() + self.frameWidth(),
            self.frameWidth(),
            self.columnWidth(0),
            self.viewport().height()
        )

    def __update_scroll_view_width(self, column: int, _: int, width: int):
        if column == 0:
            self.__scroll_view.setColumnWidth(column, width)
            self.__update_scroll_view_geometry()

    def __update_scroll_view_height(self, row: int, _: int, height: int):
        self.__scroll_view.setRowHeight(row, height)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        self.update_geometry()

    def update_geometry(self):
        self.__update_scroll_view_columns()
        self.__update_scroll_view_geometry()

    def startDrag(self, actions: Qt.DropAction):
        self.on_drag_start(actions)

    def on_drag_start(self, actions: Qt.DropAction):
        raise NotImplementedError()


if __name__ == "__main__":
    from AnyQt.QtWidgets import QMainWindow, QApplication


    class _Model(QAbstractTableModel):
        def __init__(self, data):
            super().__init__()
            self._data = data
            self._roles = [[{} for i in row] for row in data]

        def columnCount(self, _=None) -> int:
            return len(self._data[0]) if len(self._data) > 0 else 0

        def rowCount(self, _=None) -> int:
            return len(self._data)

        def data(self, index, role):
            if not index.isValid():
                return None
            if role == HeaderRole:
                return self._roles[index.row()][index.column()].get(role)
            if role == Qt.DisplayRole:
                return self._data[index.row()][index.column()]
            return None

        def setData(self, index, data, role=None):
            if not index.isValid():
                return None
            if role == HeaderRole:
                self._roles[index.row()][index.column()][role] = data
            super().setData(index, data, role)


    app = QApplication([])

    mod = _Model([["foo bar foo baz foo bar baz", "AAAA", "BBBB"]] * 20)

    # try FrozenHeaderTableView
    view = FrozenHeaderTableView()
    view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    view.set_model(mod, 3)
    view.setWordWrap(True)
    view.resizeRowsToContents()

    # try ScrollableColumnTableView
    view = ScrollableColumnTableView(
        selectionMode=QTableView.SingleSelection,
        selectionBehavior=QTableView.SelectRows,
        defaultDropAction=Qt.MoveAction,
        dragDropMode=QTableView.InternalMove,
        dragDropOverwriteMode=False,
        showGrid=False,
    )
    view.setModel(mod)
    header = view.horizontalHeader()
    header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
    header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

    win = QMainWindow()
    win.setCentralWidget(view)
    win.show()

    app.exec_()
