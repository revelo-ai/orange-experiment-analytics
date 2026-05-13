import pickle

from AnyQt.QtCore import QSize
from AnyQt.QtWidgets import QGroupBox, QLabel, QScrollArea, QVBoxLayout, QWidget
from Orange.data import Table
from Orange.widgets.utils.save.owsavebase import OWSaveBase
from orangewidget import gui
from orangewidget.utils.signals import Input
from orangewidget.widget import Msg

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    ComputeValueTransform,
    InfoTransform,
    add_transformation_to_data,
    TransformationPreprocessorList,
)

NO_DATA_MESSAGE = (
    "Data do not include transformations. It happens when the Initialize "
    "Transformation widget is not present at the beginning of the transformation "
    "workflow or when the workflow has an unsupported transformation."
)


class STLabel(QLabel):
    def __init__(self, *args):
        super().__init__(*args)
        self.setWordWrap(True)  # label wraps text if too long for the widget


class STScrollArea(QScrollArea):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(scroll_widget)
        self.scroll_layout.addStretch()  # add spacer to end to prevent labels extension
        self.setWidgetResizable(True)
        self.setWidget(scroll_widget)

    def add_labels(self, texts):
        self.clear_labels()
        for i, text in enumerate(texts):
            widget = QGroupBox()
            layout = QVBoxLayout(widget)
            layout.addWidget(STLabel(text))
            # insert widget before spacer
            self.scroll_layout.insertWidget(i, widget)

    def clear_labels(self):
        """Remove all items from the layout except spcer"""
        for i in reversed(range(self.scroll_layout.count() - 1)):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                self.scroll_layout.removeWidget(widget)
                widget.deleteLater()


class OWSaveTransformations(OWSaveBase):
    name = "Save Transformations"
    description = "Save transformations applied to the table in the pickle file"
    icon = "icons/savetransformations.svg"
    keywords = "save transformations, export"
    resizing_enabled = True

    filters = ["Pickled tranformations (*.ptr)"]

    class Inputs:
        data = Input("Data", Table)

    class Error(OWSaveBase.Error):
        no_transformations = Msg(NO_DATA_MESSAGE)

    class Warning(OWSaveBase.Warning):
        len_changed = Msg(
            "The number of data instances changed in the transformation workflow. "
            "Workflow may include the transformation that cannot be saved."
        )

    def __init__(self):
        vbox = gui.vBox(self.controlArea, "Transformations", minimumSize=(300, 300))
        self.scroll_area = STScrollArea(vbox)
        vbox.layout().addWidget(self.scroll_area)
        super().__init__(1)

    @Inputs.data
    def dataset(self, data):
        if data and TRANSFORMATIONS_ATTRIBUTE in data.attributes:
            # add domain if Orange transformation happened after last Experiment Analytics transform
            data = data.copy()
            add_transformation_to_data(data, None, data)

        self.data = data
        self.on_new_input()

    def do_save(self):
        if self.data and TRANSFORMATIONS_ATTRIBUTE in self.data.attributes:
            pl = TransformationPreprocessorList(
                self.data.attributes[TRANSFORMATIONS_ATTRIBUTE]
            )
            with open(self.filename, "wb") as f:
                pickle.dump(pl, f)

    def update_messages(self):
        super().update_messages()
        self.Warning.len_changed.clear()
        self.Error.no_transformations.clear()

        if self.data:
            # show error if table doesn't contain transformations
            no_trans = TRANSFORMATIONS_ATTRIBUTE not in self.data.attributes
            self.Error.no_transformations(shown=no_trans)
            # show warning that number of data instances changed during workflow
            self.Warning.len_changed(shown=self.data_length_changed())

        # show list of transformations
        self.list_transformations()

    def data_length_changed(self) -> bool:
        preprocessors = self.data.attributes.get(TRANSFORMATIONS_ATTRIBUTE, [])
        for trans1, trans2 in zip(preprocessors, preprocessors[1:]):
            if trans1.rows_after != trans2.rows_before:
                return True
        if preprocessors and preprocessors[-1].rows_after != len(self.data):
            # check if number of lines changed after the last transformation
            return True
        return False

    def list_transformations(self):
        if self.data:
            if TRANSFORMATIONS_ATTRIBUTE in self.data.attributes:
                self.scroll_area.add_labels(self.get_transformations())
            else:
                self.scroll_area.add_labels([NO_DATA_MESSAGE])
        else:
            self.scroll_area.add_labels(["No data on input"])

    def get_transformations(self):
        # skip first transformation - a domain saved by init widget
        tr = self.data.attributes[TRANSFORMATIONS_ATTRIBUTE][1:]
        return [str(transformation) for transformation in tr]

    def send_report(self):
        self.report_items((("File name", self.filename or "not set"),))
        if self.data and TRANSFORMATIONS_ATTRIBUTE in self.data.attributes:
            trans = [str(tr) for tr in self.get_transformations()]
            self.report_raw("Transformations", "<hr>".join(trans))

    def sizeHint(self) -> QSize:
        return QSize(400, 550)


if __name__ == "__main__":
    from orangewidget.utils.widgetpreview import WidgetPreview
    from Orange.preprocess.transformation import Identity
    from Orange.data import ContinuousVariable, Domain

    from orangecontrib.experiment_analytics.widgets.owaggregate import AggregatePreprocessor
    from orangecontrib.experiment_analytics.widgets.owslicer import SlicerPreprocessor

    data = Table("iris")

    d = data.domain
    ap = AggregatePreprocessor(
        [d["iris"]],
        [],
        [d["petal length"], d["petal width"]],
        [("Sum", "sum")],
        d["petal length"],
    )
    data = ap(data)
    ap.domain = data.domain

    d = data.domain
    sl = SlicerPreprocessor(
        d["petal length - Sum"], [((0, 100), "S1"), ((100, 8), "S2")]
    )
    data = sl(data)
    sl.domain = d

    domain = data.domain
    new_var = ContinuousVariable(
        "foo", compute_value=Identity(domain["petal length - Sum"])
    )
    new_domain = Domain([new_var] + list(domain.attributes)[1:], domain.class_vars)
    new_data = data.transform(new_domain)
    cvt = ComputeValueTransform(new_domain, domain)

    new_data.attributes[TRANSFORMATIONS_ATTRIBUTE] = [InfoTransform(d), sl, ap, cvt]

    WidgetPreview(OWSaveTransformations).run(new_data)
