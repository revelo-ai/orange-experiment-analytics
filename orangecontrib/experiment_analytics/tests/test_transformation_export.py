import unittest
from datetime import datetime, timezone
from importlib.metadata import version

import numpy as np
import pandas as pd
from Orange.preprocess.transformation import Identity
from numpy.testing import assert_array_equal
from Orange.data import (
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    Table,
    StringVariable,
    TimeVariable,
)
from Orange.preprocess import Normalize

from orangecontrib.experiment_analytics.transformation_export import (
    TRANSFORMATIONS_ATTRIBUTE,
    ComputeValueTransform,
    InfoTransform,
    add_transformation_to_data,
    TransformationPreprocessorList,
    HTML_TABLE_STYLE,
)
from orangecontrib.experiment_analytics.widgets.owslicer import SlicerPreprocessor


class TestAddTransformation(unittest.TestCase):
    def setUp(self):
        self.data = Table("iris")
        cvt = InfoTransform(self.data.domain)
        cvt.set_row_count(self.data, self.data)
        self.data.attributes[TRANSFORMATIONS_ATTRIBUTE] = (cvt,)
        s = [((0.1, 2.2), "Slice 1"), ((2.3, 5), "Slice 2")]
        self.transformation = SlicerPreprocessor(self.data.domain["petal length"], s)

    def test_no_init(self):
        """Test TRANSFORMATIONS_ATTRIBUTE not set - do not do anything"""
        self.data.attributes.pop(TRANSFORMATIONS_ATTRIBUTE)
        add_transformation_to_data(self.data, self.transformation, self.data)
        self.assertNotIn(TRANSFORMATIONS_ATTRIBUTE, self.data.attributes)

    def test_domain_same(self):
        """
        Insert only transformation, domain haven't changed from previous transformation
        """
        data = self.transformation(self.data)
        add_transformation_to_data(data, self.transformation, self.data)
        trans = data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(trans))
        self.assertEqual(self.transformation, trans[1])

        self.assertEqual(len(self.data), trans[0].rows_before)
        self.assertEqual(len(self.data), trans[0].rows_after)
        self.assertEqual(len(self.data), trans[1].rows_before)
        self.assertEqual(108, trans[1].rows_after)

    def test_domain_changed(self):
        """Insert domain preprocessor and transformation, domain have changed"""
        data_before = Normalize()(self.data)
        s = [((0.1, 2.2), "Slice 1"), ((2.3, 5), "Slice 2")]
        transformation = SlicerPreprocessor(data_before.domain["petal length"], s)
        new_data = transformation(data_before)
        add_transformation_to_data(new_data, transformation, data_before)
        trans = new_data.attributes[TRANSFORMATIONS_ATTRIBUTE]

        self.assertEqual(3, len(trans))
        self.assertIsInstance(trans[1], ComputeValueTransform)
        self.assertEqual(data_before.domain, trans[1].domain)
        self.assertEqual(transformation, trans[2])

        self.assertEqual(len(self.data), trans[0].rows_before)
        self.assertEqual(len(self.data), trans[0].rows_after)
        self.assertEqual(len(self.data), trans[1].rows_before)
        self.assertEqual(len(self.data), trans[1].rows_after)
        self.assertEqual(len(self.data), trans[2].rows_before)
        self.assertEqual(89, trans[2].rows_after)

    def test_only_domain(self):
        new_data = Normalize()(self.data)
        add_transformation_to_data(new_data, None, new_data)
        trans = new_data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        self.assertEqual(2, len(trans))
        self.assertIsInstance(trans[1], ComputeValueTransform)
        self.assertEqual(new_data.domain, trans[1].domain)

        self.assertEqual(len(self.data), trans[0].rows_before)
        self.assertEqual(len(self.data), trans[0].rows_after)
        self.assertEqual(len(self.data), trans[1].rows_before)
        self.assertEqual(len(self.data), trans[1].rows_after)


class TestComputeValueTransform(unittest.TestCase):
    def setUp(self):
        self.data = Table("iris")

    def test_no_compute_value(self):
        cvt = ComputeValueTransform(self.data.domain, self.data.domain)
        result = cvt(self.data)
        self.assertEqual(self.data.domain, result.domain)
        assert_array_equal(self.data.X, result.X)
        assert_array_equal(self.data.Y, result.Y)

    def test_with_compute_value(self):
        normalized_data = Normalize()(self.data)
        cvt = ComputeValueTransform(normalized_data.domain, self.data.domain)
        result = cvt(self.data)
        # resulting array should be same than normalized_data
        self.assertEqual(normalized_data.domain, result.domain)
        assert_array_equal(normalized_data.X, result.X)
        assert_array_equal(normalized_data.Y, result.Y)
        # it should be different to the original data
        self.assertFalse(np.array_equal(self.data.X, result.X))

    def test_different_data(self):
        normalized_data = Normalize()(self.data)
        cvt = ComputeValueTransform(normalized_data.domain, self.data.domain)
        result = cvt(Table("housing"))
        self.assertEqual(normalized_data.domain, result.domain)

    def test_repr_changed(self):
        normalized_data = Normalize()(self.data)
        cvt = ComputeValueTransform(normalized_data.domain, self.data.domain)
        self.assertEqual(
            f"<h4>Domain transformation</h4>{HTML_TABLE_STYLE}"
            "<div><table><tr><th>Changed/added features: </th><td>4</td></tr>"
            "<tr><th>Unchanged features: </th><td>1</td></tr></table></div>",
            str(cvt),
        )

    def test_repr_no_compute_value(self):
        cvt = ComputeValueTransform(self.data.domain, self.data.domain)
        self.assertEqual(
            f"<h4>Domain transformation</h4>{HTML_TABLE_STYLE}<div><table>"
            "<tr><th>Unchanged features: </th><td>5</td></tr></table></div>",
            str(cvt),
        )

    def test_repr_removed(self):
        d = self.data.domain
        transformed_data = self.data.transform(Domain(d.attributes[1:], d.class_vars))
        cvt = ComputeValueTransform(transformed_data.domain, self.data.domain)
        self.assertEqual(
            f"<h4>Domain transformation</h4>{HTML_TABLE_STYLE}"
            "<div><table><tr><th>Removed features: </th><td>1</td></tr>"
            "<tr><th>Unchanged features: </th><td>4</td></tr></table></div>",
            str(cvt),
        )

        transformed_data = self.data.transform(Domain(d.attributes, []))
        cvt = ComputeValueTransform(transformed_data.domain, self.data.domain)
        self.assertEqual(
            f"<h4>Domain transformation</h4>{HTML_TABLE_STYLE}"
            "<div><table><tr><th>Removed features: </th><td>1</td></tr>"
            "<tr><th>Unchanged features: </th><td>4</td></tr></table></div>",
            str(cvt),
        )

    def test_repr_moved(self):
        d = self.data.domain
        class_var = d.class_vars + d.attributes[:2]
        transformed_data = self.data.transform(Domain(d.attributes[2:], class_var))
        cvt = ComputeValueTransform(transformed_data.domain, self.data.domain)
        self.assertEqual(
            f"<h4>Domain transformation</h4>{HTML_TABLE_STYLE}"
            "<div><table><tr><th>Moved features: </th><td>2</td></tr>"
            "<tr><th>Unchanged features: </th><td>3</td></tr></table></div>",
            str(cvt),
        )

        transformed_data = self.data.transform(Domain(d.attributes + d.class_vars, []))
        cvt = ComputeValueTransform(transformed_data.domain, self.data.domain)
        self.assertEqual(
            f"<h4>Domain transformation</h4>{HTML_TABLE_STYLE}"
            "<div><table><tr><th>Moved features: </th><td>1</td></tr>"
            "<tr><th>Unchanged features: </th><td>4</td></tr></table></div>",
            str(cvt),
        )


class TestInfoTransform(unittest.TestCase):
    def setUp(self):
        self.data = Table("iris")

    def test_info_transform_same_domain(self):
        info = InfoTransform(self.data.domain)
        result = info(self.data)
        self.assertEqual(self.data.domain, result.domain)
        assert_array_equal(self.data.X, result.X)
        assert_array_equal(self.data.Y, result.Y)

    def test_info_transform_missing_features(self):
        info = InfoTransform(self.data.domain)

        # data with missing class and one feature
        data = self.data.transform(Domain(self.data.domain.attributes[:-1]))
        with self.assertRaises(ValueError) as err:
            info(data)
        self.assertEqual(
            "The data are missing the following features: petal width, iris",
            str(err.exception),
        )

        # data with missing class
        data = self.data.transform(Domain(self.data.domain.attributes))
        with self.assertRaises(ValueError) as err:
            info(data)
        self.assertEqual(
            "The data are missing the following features: iris", str(err.exception)
        )

        # data with missing class
        d = self.data.domain
        data = self.data.transform(Domain(d.attributes[1:], d.class_vars))
        with self.assertRaises(ValueError) as err:
            info(data)
        self.assertEqual(
            "The data are missing the following features: sepal length",
            str(err.exception),
        )

    def test_info_transform_extra_features(self):
        info = InfoTransform(self.data.domain)

        # extra feature in attributes
        domain = self.data.domain
        var = (ContinuousVariable("a"),)
        data = self.data.transform(Domain(domain.attributes + var, domain.class_vars))
        with self.assertWarns(UserWarning) as warn:
            result = info(data)
        # table transformed to original domain
        self.assertEqual(self.data.domain, result.domain)
        assert_array_equal(self.data.X, result.X)
        assert_array_equal(self.data.Y, result.Y)
        self.assertEqual(
            "The data includes the following features that haven't been present"
            " in the transformation workflow data: a",
            str(warn.warning),
        )

        # extra feature in class_vars
        domain = self.data.domain
        data = self.data.transform(Domain(domain.attributes, domain.class_vars + var))
        with self.assertWarns(UserWarning) as warn:
            result = info(data)
        # table transformed to original domain
        self.assertEqual(self.data.domain, result.domain)
        assert_array_equal(self.data.X, result.X)
        assert_array_equal(self.data.Y, result.Y)
        self.assertEqual(
            "The data includes the following features that haven't been present"
            " in the transformation workflow data: a",
            str(warn.warning),
        )

        # extra feature in metas
        domain = self.data.domain
        data = self.data.transform(Domain(domain.attributes, domain.class_vars, var))
        with self.assertWarns(UserWarning) as warn:
            result = info(data)
        # table transformed to original domain
        self.assertEqual(self.data.domain, result.domain)
        assert_array_equal(self.data.X, result.X)
        assert_array_equal(self.data.Y, result.Y)
        self.assertEqual(
            "The data includes the following features that haven't been present"
            " in the transformation workflow data: a",
            str(warn.warning),
        )

    def test_repr(self):
        info = InfoTransform(self.data.domain)
        self.assertEqual("Domain info", str(info))


class TestTransformationPreprocessorList(unittest.TestCase):
    """Test Orange preprocessor list to identify potential changes"""

    def setUp(self):
        domain = Domain([ContinuousVariable("a"), ContinuousVariable("b")])
        x = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 3], [2, 5]]
        self.data = Table.from_list(domain, x)

        self.df = pd.DataFrame({"a": [2, 1, 4, 2], "b": [3, 4, 5, 2]})

    def test_simple_list(self):
        a_var = self.data.domain[0]
        pps = [
            InfoTransform(self.data.domain),
            SlicerPreprocessor(a_var, [((1, 2.1), "S1"), ((2.5, 6), "S2")]),
        ]
        result = TransformationPreprocessorList(pps)(self.data)

        self.assertTupleEqual(self.data.domain.attributes, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        expected = np.array([[1, 2], [2, 3], [1, 3], [2, 5], [3, 4], [4, 5]])
        np.testing.assert_array_equal(expected, result.X)
        expected = np.array([[0], [0], [0], [0], [1], [1]])
        np.testing.assert_array_equal(expected, result.metas)

        # test with pandas
        result_pd = TransformationPreprocessorList(pps).from_pandas(self.df)
        self.assertTupleEqual(self.data.domain.attributes, result_pd.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result_pd.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result_pd.domain.metas[0].values)

        expected = np.array([[2, 3], [1, 4], [2, 2], [4, 5]])
        np.testing.assert_array_equal(expected, result_pd.X)
        np.testing.assert_array_equal(np.array([[0], [0], [0], [1]]), result_pd.metas)

    def test_with_domain_transform(self):
        new_domain = Normalize(norm_type=Normalize.NormalizeBySpan)(self.data).domain
        a_var = new_domain[0]

        pps = [
            InfoTransform(self.data.domain),
            ComputeValueTransform(new_domain, self.data.domain),
            SlicerPreprocessor(a_var, [((0, 0.5), "S1"), ((0.5, 1), "S2")]),
        ]
        result = TransformationPreprocessorList(pps)(self.data)

        self.assertTupleEqual(new_domain.attributes, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        expected = np.array(
            [[0, 0], [0.333, 0.333], [0, 0.333], [0.333, 1], [0.666, 0.666], [1, 1]]
        )
        np.testing.assert_array_almost_equal(expected, result.X, decimal=3)
        expected = np.array([[0], [0], [0], [0], [1], [1]])
        np.testing.assert_array_equal(expected, result.metas)

    def test_with_domain_transform2(self):
        a_var = self.data.domain[0]
        slicer = SlicerPreprocessor(a_var, [((1, 2.1), "S1"), ((2.5, 6), "S2")])
        s_data = slicer(self.data)
        n_domain = Normalize(norm_type=Normalize.NormalizeBySpan)(s_data).domain

        pps = [slicer, ComputeValueTransform(n_domain, s_data.domain)]
        result = TransformationPreprocessorList(pps)(self.data)

        self.assertTupleEqual(n_domain.attributes, result.domain.attributes)
        self.assertTupleEqual((DiscreteVariable("Slice"),), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[0].values)

        expected = np.array(
            [[0, 0], [0.333, 0.333], [0, 0.333], [0.333, 1], [0.666, 0.666], [1, 1]]
        )
        np.testing.assert_array_almost_equal(expected, result.X, decimal=3)
        expected = np.array([[0], [0], [0], [0], [1], [1]])
        np.testing.assert_array_equal(expected, result.metas)

    def test_fail(self):
        var = ContinuousVariable("c")
        pps = [SlicerPreprocessor(var, [((1, 2.1), "S1"), ((2.5, 6), "S2")])]
        with self.assertRaises(ValueError) as err:
            TransformationPreprocessorList(pps)(self.data)
        self.assertEqual(
            "The Series Slicer transformation expects data to contain c variable, "
            "which is missing in the data.",
            str(err.exception),
        )

    def test_df_to_table(self):
        c = ContinuousVariable("cont")
        d = DiscreteVariable("disc")
        t = TimeVariable("time")
        s = StringVariable("str")
        domain = Domain([c, d, t], metas=[s])
        columns = {
            "cont": [1, 2, 3, 4],
            "disc": list("abcd"),
            "str": list("efgh"),
            "time": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
        df = pd.DataFrame(columns)

        pl = TransformationPreprocessorList([InfoTransform(domain)])
        res_table = pl._TransformationPreprocessorList__df_to_table(df)
        self.assertTupleEqual((c, d, t), res_table.domain.attributes)
        self.assertTupleEqual((), res_table.domain.class_vars)
        self.assertTupleEqual((s,), res_table.domain.metas)

        exp = [
            [1, 0, datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()],
            [2, 1, datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()],
            [3, 2, datetime(2024, 1, 3, tzinfo=timezone.utc).timestamp()],
            [4, 3, datetime(2024, 1, 4, tzinfo=timezone.utc).timestamp()],
        ]
        assert_array_equal(np.array(exp), res_table.X)

    def test_df_to_table_exception(self):
        domain = Domain([ContinuousVariable("cont")])
        df = pd.DataFrame({"cont": [1, 2, 3, "a"]})

        pl = TransformationPreprocessorList([InfoTransform(domain)])
        with self.assertRaises(ValueError) as err:
            pl._TransformationPreprocessorList__df_to_table(df)
        exp = "Converting cont feature to float failed with:"
        self.assertTrue(str(err.exception).startswith(exp))

        domain = Domain([TimeVariable("time")])
        df = pd.DataFrame({"time": ["2024-01-01", "2024-01-02", "2024-01-03", "b"]})

        pl = TransformationPreprocessorList([InfoTransform(domain)])
        with self.assertRaises(ValueError) as err:
            pl._TransformationPreprocessorList__df_to_table(df)
        exp = "Converting time feature to datetime64[ns] failed with:"
        self.assertTrue(str(err.exception).startswith(exp))

    def test_from_pandas(self):
        columns = {
            "cont": [1, 2, 3, 4],
            "disc": list("abcc"),
            "str": list("eggh"),
            "time": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-01"],
        }
        df = pd.DataFrame(columns)

        c, d = ContinuousVariable("cont"), DiscreteVariable("disc", values=list("bac"))
        s, t = StringVariable("str"), TimeVariable("time")
        t_tr = ContinuousVariable("time", compute_value=Identity(t))
        initial_domain = Domain([c, d, t], metas=[s])
        transformed_domain = Domain([c, t_tr], metas=[s, d])
        pps = [
            InfoTransform(initial_domain),
            ComputeValueTransform(transformed_domain),
            SlicerPreprocessor(c, [((1, 2.1), "S1"), ((2.5, 6), "S2")]),
        ]
        result = TransformationPreprocessorList(pps).from_pandas(df)

        self.assertTupleEqual((c, t_tr), result.domain.attributes)
        self.assertTupleEqual((s, d, DiscreteVariable("Slice")), result.domain.metas)
        self.assertTupleEqual(("S1", "S2"), result.domain.metas[2].values)

        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
        t2 = datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp()
        expected = np.array([[1, t1], [2, t2], [3, t1], [4, t1]])
        np.testing.assert_array_equal(expected, result.X)

        expected = np.array(
            [["e", 1.0, 0], ["g", 0.0, 0], ["g", 2.0, 1], ["h", 2.0, 1]],
            dtype=object,
        )
        np.testing.assert_array_equal(expected, result.metas)

    def test_requirements(self):
        pps = [InfoTransform(self.data.domain)]
        transformation = TransformationPreprocessorList(pps)
        req = transformation.requirements
        exp = {
            "numpy": version('numpy'),
            "Orange3": version('Orange3'),
            "orange-experiment-analytics": version('Orange-experiment_analytics'),
            "pandas": version('pandas'),
            "scikit-learn": version('scikit-learn'),
            "scipy": version('scipy'),

        }
        # test subset (keys form exp) matches, req can contain also other orange addons
        self.assertEqual(req, req | exp)


if __name__ == "__main__":
    unittest.main()
