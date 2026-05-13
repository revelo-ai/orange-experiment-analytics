import unittest

import numpy as np
from Orange.data import Table, ContinuousVariable, Domain
from Orange.preprocess import Normalize
from Orange.preprocess.transformation import Normalizer, Identity
from Orange.widgets.tests.base import WidgetTest
from numpy.testing import assert_array_equal

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    InfoTransform,
)
from orangecontrib.experiment_analytics.widgets.owinitializetransformation import (
    OWInitializeTransformation,
)


class TestOWLetterReport(WidgetTest):
    def setUp(self):
        self.data = Table("brown-selected")
        self.widget = self.create_widget(OWInitializeTransformation)

    def test_data(self):
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

        self.send_signal(self.widget.Inputs.data, self.data)
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(self.data.domain, output.domain)
        np.testing.assert_array_equal(self.data.X, output.X)
        np.testing.assert_array_equal(self.data.metas, output.metas)
        trans = output.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(1, len(trans))
        self.assertIsInstance(trans[0], InfoTransform)
        self.assertEqual(output.domain, trans[0].domain)
        self.assertNotIn(TRANSFORMATIONS_ATTRIBUTE, self.data.attributes)

        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.data))

    def test_no_compute_values(self):
        data = Normalize()(self.data)
        data.domain.metas[0]._compute_value = "Test"
        data.domain.class_vars[0]._compute_value = "Test1"

        self.assertIsInstance(data.domain["alpha 0"].compute_value, Normalizer)
        self.assertIsInstance(data.domain["alpha 7"].compute_value, Normalizer)
        self.assertIsInstance(data.domain["Elu 150"].compute_value, Normalizer)
        self.assertIsInstance(data.domain["spo 2"].compute_value, Normalizer)
        self.assertIsInstance(data.domain["function"].compute_value, str)
        self.assertIsInstance(data.domain["gene"].compute_value, str)

        self.send_signal(self.widget.Inputs.data, self.data)
        output = self.get_output(self.widget.Outputs.data)

        self.assertIsNone(output.domain["alpha 0"].compute_value)
        self.assertIsNone(output.domain["alpha 7"].compute_value)
        self.assertIsNone(output.domain["Elu 150"].compute_value)
        self.assertIsNone(output.domain["spo 2"].compute_value)
        self.assertIsNone(output.domain["function"].compute_value)
        self.assertIsNone(output.domain["gene"].compute_value)

    def test_compute_value_before_init(self):
        # create iris with renamed first variable
        iris = Table("iris")
        domain = iris.domain
        cv = Identity(variable=domain["sepal length"])
        new_sl = ContinuousVariable("foo", compute_value=cv)
        new_domain = Domain([new_sl] + list(domain.attributes)[1:], domain.class_vars)
        new_iris = iris.transform(new_domain)
        assert_array_equal(iris.X, new_iris.X)
        assert_array_equal(iris.Y, new_iris.Y)

        # init widget should keep all values in data
        self.send_signal(self.widget.Inputs.data, new_iris)
        output = self.get_output(self.widget.Outputs.data)
        assert_array_equal(iris.X, output.X)
        assert_array_equal(iris.Y, output.Y)


if __name__ == "__main__":
    unittest.main()
