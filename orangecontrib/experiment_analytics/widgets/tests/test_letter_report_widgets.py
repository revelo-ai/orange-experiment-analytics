import unittest

from AnyQt.QtCore import QAbstractTableModel, Qt

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.experiment_analytics.widgets.letter_report_widgets import \
    FrozenHeaderTableView, ScrollableColumnTableView


class _Model(QAbstractTableModel):
    def __init__(self, data):
        super().__init__()
        self._data = data

    def columnCount(self, _=None) -> int:
        return len(self._data[0]) if len(self._data) > 0 else 0

    def rowCount(self, _=None) -> int:
        return len(self._data)

    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            return self._data[index.row()][index.column()]
        return None


class TestFrozenHeaderTableView(WidgetTest):
    def setUp(self):
        self.n_all_rows = 10
        self.model = _Model([["foo", "bar", "baz"]] * self.n_all_rows)
        self.view = FrozenHeaderTableView()

    def test_set_model(self):
        n_header_rows = 5
        self.view.set_model(self.model, n_header_rows)
        self.assertEqual(self.view.model().rowCount(), self.n_all_rows)
        self.assertEqual(self.view.n_header_rows, n_header_rows)


class TestScrollableColumnTableView(WidgetTest):
    def setUp(self):
        self.n_all_rows = 10
        self.model = _Model([["foo", "bar", "baz"]] * self.n_all_rows)
        self.view = ScrollableColumnTableView()

    def test_set_model(self):
        self.view.setModel(self.model)
        self.assertEqual(self.view.model().rowCount(), self.n_all_rows)


if __name__ == "__main__":
    unittest.main()
