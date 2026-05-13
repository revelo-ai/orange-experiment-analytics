import os

import xlsxwriter
from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QFileDialog, QWidget

from Orange.widgets import gui
from Orange.widgets.utils.itemmodels import PyTableModel

BorderRole = next(gui.OrangeUserRole)


def export(model: PyTableModel, n_header: int, path: str):
    workbook = xlsxwriter.Workbook(path)
    worksheet = workbook.add_worksheet("Sheet 1")
    worksheet.freeze_panes(n_header, 0)
    worksheet.set_column(0, 0, width=30)
    worksheet.set_column(1, model.columnCount() - 1, width=20)

    horizontal_vertical_header_format = workbook.add_format({
        "align": "center",
        "right": 1,
        "text_wrap": True
    })
    horizontal_header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "text_wrap": True,
    })
    right_horizontal_header_format = workbook.add_format({
        "bold": True,
        "align": "center",
        "text_wrap": True,
        "right": 1,
    })
    border_vertical_header_format = workbook.add_format({
        "align": "left",
        "top": 1,
        "right": 1,
        "text_wrap": True
    })
    vertical_header_format = workbook.add_format({
        "align": "left",
        "right": 1,
        "text_wrap": True
    })
    bottom_border_vertical_header_format = workbook.add_format({
        "align": "left",
        "top": 1,
        "bottom": 1,
        "right": 1,
        "text_wrap": True
    })
    bottom_vertical_header_format = workbook.add_format({
        "align": "left",
        "bottom": 1,
        "right": 1,
        "text_wrap": True
    })
    total_header_format = workbook.add_format({
        "align": "left",
        "bold": True,
        "italic": True,
        "bottom": 1,
        "top": 1,
        "right": 1,
        "text_wrap": True
    })
    total_format = workbook.add_format({
        "align": "center",
        "italic": True,
        "bottom": 1,
        "top": 1,
        "text_wrap": True
    })
    right_total_format = workbook.add_format({
        "align": "center",
        "italic": True,
        "bottom": 1,
        "top": 1,
        "right": 1,
    })
    center_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
    })
    right_center_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
        "right": 1,
    })
    border_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
        "top": 1,
    })
    right_border_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
        "top": 1,
        "right": 1,
    })
    bottom_center_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
        "bottom": 1,
    })
    right_bottom_center_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
        "bottom": 1,
        "right": 1,
    })
    bottom_border_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
        "top": 1,
        "bottom": 1,
    })
    right_bottom_border_format = workbook.add_format({
        "align": "center",
        "text_wrap": True,
        "top": 1,
        "bottom": 1,
        "right": 1,
    })

    n_rows, n_columns = model.rowCount(), model.columnCount()
    for i in range(n_rows):
        for j in range(n_columns):

            data = model.data(model.index(i, j), role=Qt.DisplayRole)
            border = model.data(model.index(i, j), role=BorderRole)

            if i == n_header - 1 and j == n_columns - 1:
                cell_format = right_total_format
            elif i == n_header - 1:
                cell_format = total_header_format if j == 0 else total_format
            elif j == 0 and i < n_header:
                cell_format = horizontal_vertical_header_format
            elif i < n_header and j == n_columns - 1:
                cell_format = right_horizontal_header_format
            elif i < n_header:
                cell_format = horizontal_header_format
            elif j == 0 and i == n_rows - 1:
                cell_format = bottom_border_vertical_header_format \
                    if border else bottom_vertical_header_format
            elif j == 0:
                cell_format = border_vertical_header_format \
                    if border else vertical_header_format
            elif i == n_rows - 1 and j == n_columns - 1:
                cell_format = right_bottom_border_format \
                    if border else right_bottom_center_format
            elif i == n_rows - 1:
                cell_format = bottom_border_format \
                    if border else bottom_center_format
            elif j == n_columns - 1:
                cell_format = right_border_format \
                    if border else right_center_format
            else:
                cell_format = border_format if border else center_format

            worksheet.write_string(i, j, data, cell_format)

    workbook.close()


def save(widget: QWidget, model: PyTableModel, n_rows: int):
    filename, _ = QFileDialog.getSaveFileName(
        widget, "Save", os.path.expanduser("~/"),
        "Microsoft Excel spreadsheet (*.xlsx)"
    )
    if filename:
        export(model, n_rows, filename)
