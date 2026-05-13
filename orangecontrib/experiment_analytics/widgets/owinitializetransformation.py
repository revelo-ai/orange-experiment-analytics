from typing import Optional

from Orange.data import Domain, Table
from Orange.widgets.widget import OWWidget
from orangewidget import gui
from orangewidget.utils.signals import Input, Output

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    InfoTransform,
)


class OWInitializeTransformation(OWWidget):
    name = "Initialize Transformation"
    description = "Mark beginning of transformations pipeline that will be exported."
    icon = "icons/inittransformation.svg"
    keywords = ["initialize", "export", "transformation"]
    priority = 310

    want_control_area = False

    class Inputs:
        data = Input("Data", Table)

    class Outputs:
        data = Output("Data", Table)

    def __init__(self):
        gui.label(
            self.mainArea,
            self,
            "This widget marks the beginning of \n"
            "the data transformation pipeline \n"
            "that will be exported by the \n"
            "Save Transformations widget.",
            margin=10,
        )

    @Inputs.data
    def set_data(self, data: Optional[Table]):
        self.commit(data)

    def commit(self, data):
        if data:
            # reset comput values on variables that preprocessing before this
            # widget doesn't affect preprocessing pipeline
            new_domain = Domain(
                attributes=[a.copy(compute_value=None) for a in data.domain.attributes],
                class_vars=[a.copy(compute_value=None) for a in data.domain.class_vars],
                metas=[a.copy(compute_value=None) for a in data.domain.metas],
            )
            data = data.from_numpy(
                domain=new_domain,
                X=data.X,
                Y=data.Y,
                metas=data.metas,
                W=data.W,
                attributes=data.attributes.copy(),
                ids=data.ids,
            )
            # store domain to identify Orange transforms between init and next Experiment Analytics widget
            info = InfoTransform(data.domain)
            info.set_row_count(data, data)
            data.attributes[TRANSFORMATIONS_ATTRIBUTE] = (info,)
        self.Outputs.data.send(data)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    table_ = Table("iris")
    WidgetPreview(OWInitializeTransformation).run(table_)
