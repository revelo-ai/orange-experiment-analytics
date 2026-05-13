import unittest
from typing import Iterable, Set
from unittest import TestCase
from unittest.mock import Mock

from Orange.classification import LogisticRegressionLearner, RandomForestLearner
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table, Variable
from Orange.evaluation import Results, TestOnTrainingData
from Orange.modelling import ConstantLearner
from Orange.regression import LinearRegressionLearner

from orangecontrib.experiment_analytics.stepwise_feature_selection import (
    Constant0Learner,
    FeatureSelectionPreprocessor,
    Scoring,
    StepwiseFeatureSelection,
    Stopping,
    _filter_data,
    _predict,
)


def ft_set(features: Iterable[Variable]) -> Set[str]:
    return {a.name for a in features}


class TestUtils(TestCase):
    def test_filter_data(self):
        iris = Table("iris")
        d = iris.domain
        fdata = _filter_data(iris, set(d.attributes[1:3]))
        self.assertEqual(len(iris), len(fdata))
        expected = (d["sepal width"], d["petal length"])
        self.assertTupleEqual(expected, fdata.domain.attributes)
        self.assertTupleEqual((d["iris"],), fdata.domain.class_vars)
        self.assertTupleEqual((), fdata.domain.metas)

    def test_predict(self):
        iris = Table("iris")
        res = _predict(iris, LogisticRegressionLearner(), TestOnTrainingData())
        self.assertIsInstance(res.learners[0], LogisticRegressionLearner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(iris), len(res.predicted[0]))

        learner = LogisticRegressionLearner(fit_intercept=False)
        res = _predict(iris, learner, TestOnTrainingData())
        # should use logistic regression since data have attributes
        self.assertIsInstance(res.learners[0], LogisticRegressionLearner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(iris), len(res.predicted[0]))

        iris_no_attr = iris.transform(Domain([], class_vars=iris.domain.class_vars))
        learner = LogisticRegressionLearner()
        res = _predict(iris_no_attr, learner, TestOnTrainingData())
        self.assertIsInstance(res.learners[0], ConstantLearner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(iris), len(res.predicted[0]))

        learner = LogisticRegressionLearner(fit_intercept=False)
        res = _predict(iris_no_attr, learner, TestOnTrainingData())
        self.assertIsInstance(res.learners[0], ConstantLearner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(iris), len(res.predicted[0]))

        housing = Table("housing")
        res = _predict(housing, LinearRegressionLearner(), TestOnTrainingData())
        self.assertIsInstance(res.learners[0], LinearRegressionLearner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(housing), len(res.predicted[0]))

        learner = LinearRegressionLearner(fit_intercept=False)
        res = _predict(housing, learner, TestOnTrainingData())
        # should use logistic regression since data have attributes
        self.assertIsInstance(res.learners[0], LinearRegressionLearner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(housing), len(res.predicted[0]))

        domain = Domain([], class_vars=iris.domain.class_vars)
        housing_no_attr = housing.transform(domain)
        learner = LinearRegressionLearner()
        res = _predict(housing_no_attr, learner, TestOnTrainingData())
        self.assertIsInstance(res.learners[0], ConstantLearner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(housing), len(res.predicted[0]))

        learner = LinearRegressionLearner(fit_intercept=False)
        res = _predict(housing_no_attr, learner, TestOnTrainingData())
        self.assertIsInstance(res.learners[0], Constant0Learner)
        self.assertEqual(1, len(res.predicted))
        self.assertEqual(len(housing), len(res.predicted[0]))


class TestScoring(TestCase):
    def setUp(self):
        self.iris = Table("iris")
        self.housing = Table("housing")

    def test_get_learner(self):
        sc = Scoring("AUC", ("Random split", {"test_size": 0.1}))
        self.assertIsInstance(sc.get_learner(self.iris), LogisticRegressionLearner)
        self.assertIsInstance(sc.get_learner(self.housing), LinearRegressionLearner)

        sc = Scoring("AUC", ("Random split", {"test_size": 0.1}), ConstantLearner())
        self.assertIsInstance(sc.get_learner(self.housing), ConstantLearner)

    def test_score_selection_classification(self):
        d = self.iris.domain
        sc = Scoring("CA", ("Cross validation", {"k": 5}))
        score = sc.score_selection(self.iris, set())
        self.assertAlmostEqual(0.333, score, places=3)
        self.assertIsNone(sc.baseline_evaluation)

        score = sc.score_selection(self.iris, {d["petal length"]})
        self.assertAlmostEqual(0.94, score, places=1)
        self.assertIsNone(sc.baseline_evaluation)

        score = sc.score_selection(self.iris, {d["petal length"], d["sepal length"]})
        self.assertAlmostEqual(0.95, score, places=1)
        self.assertIsNone(sc.baseline_evaluation)

        score = sc.score_selection(self.iris, {d["petal length"]}, is_baseline=True)
        self.assertAlmostEqual(0.94, score, places=1)
        self.assertIsInstance(sc.baseline_evaluation, Results)

    def test_score_selection_regression(self):
        d = self.housing.domain
        sc = Scoring("R2", ("Cross validation", {"k": 5}))
        score = sc.score_selection(self.housing, set())
        self.assertAlmostEqual(-0.002, score, places=3)
        self.assertIsNone(sc.baseline_evaluation)

        score = sc.score_selection(self.housing, {d["LSTAT"]})
        self.assertAlmostEqual(0.54, score, places=1)
        self.assertIsNone(sc.baseline_evaluation)

        score = sc.score_selection(self.housing, {d["LSTAT"], d["RM"]})
        self.assertAlmostEqual(0.63, score, places=1)
        self.assertIsNone(sc.baseline_evaluation)

        score = sc.score_selection(self.housing, {d["LSTAT"]}, is_baseline=True)
        self.assertAlmostEqual(0.54, score, places=1)
        self.assertIsInstance(sc.baseline_evaluation, Results)

    def test_score_selection_all_methods_regression(self):
        housing = self.housing
        sc = Scoring("R2", ("Cross validation", {"k": 5}))
        scores = sc.score_selection_all_methods(None)
        self.assertDictEqual({}, scores)

        # no baseline - when no features in the model - all scores should be NA
        scores = sc.score_selection_all_methods(housing)
        self.assertDictEqual({"MAE": "NA", "R2": "NA", "RMSE": "NA"}, scores)

        # compute baseline
        sc.score_selection(self.housing, {housing.domain["LSTAT"]}, is_baseline=True)
        scores = sc.score_selection_all_methods(housing)
        self.assertEqual(3, len(scores))
        self.assertAlmostEqual(0.54, scores["R2"], places=1)
        self.assertAlmostEqual(4.51, scores["MAE"], places=1)
        self.assertAlmostEqual(6.22, scores["RMSE"], places=1)

    def test_score_selection_all_methods_classification(self):
        iris = self.iris
        sc = Scoring("AUC", ("Cross validation", {"k": 5}))

        # no baseline - when no features in the model - all scores should be NA
        scores = sc.score_selection_all_methods(iris)
        expected = {"AUC": "NA", "CA": "NA", "F1": "NA", "Prec": "NA", "Recall": "NA"}
        self.assertDictEqual(expected, scores)

        # compute baseline
        sc.score_selection(self.iris, {iris.domain["petal length"]}, is_baseline=True)
        scores = sc.score_selection_all_methods(iris)
        self.assertEqual(5, len(scores))
        self.assertAlmostEqual(0.99, scores["AUC"], places=1)
        self.assertAlmostEqual(0.94, scores["CA"], places=1)
        self.assertAlmostEqual(0.95, scores["F1"], places=1)
        self.assertAlmostEqual(0.95, scores["Prec"], places=1)
        self.assertAlmostEqual(0.95, scores["Recall"], places=1)

    def test_compute_feature_scores_forward(self):
        d = self.housing.domain
        sc = Scoring("R2", ("Cross validation", {"k": 5}))
        scores = sc.compute_feature_scores(self.housing, set(), set(), "Forward")
        self.assertEqual(len(self.housing.domain.attributes), len(scores))
        self.assertAlmostEqual(0.541, scores[d["LSTAT"]], places=1)
        self.assertAlmostEqual(0.475, scores[d["RM"]], places=1)
        self.assertAlmostEqual(0.02, scores[d["CHAS"]], places=1)

        scores = sc.compute_feature_scores(self.housing, {d["LSTAT"]}, set(), "Forward")
        self.assertEqual(len(self.housing.domain.attributes) - 1, len(scores))
        self.assertNotIn(d["LSTAT"], scores)
        self.assertAlmostEqual(0.091, scores[d["RM"]], places=1)
        self.assertAlmostEqual(0.016, scores[d["CHAS"]], places=1)
        self.assertAlmostEqual(0.056, scores[d["PTRATIO"]], places=1)

        selected = {d["LSTAT"], d["RM"]}
        scores = sc.compute_feature_scores(self.housing, selected, set(), "Forward")
        self.assertEqual(len(self.housing.domain.attributes) - 2, len(scores))
        self.assertNotIn(d["LSTAT"], scores)
        self.assertNotIn(d["RM"], scores)
        self.assertAlmostEqual(0.011, scores[d["CHAS"]], places=1)
        self.assertAlmostEqual(0.036, scores[d["PTRATIO"]], places=1)

    def test_compute_feature_scores_backward(self):
        d = self.housing.domain
        sc = Scoring("R2", ("Cross validation", {"k": 5}))
        scores = sc.compute_feature_scores(self.housing, set(), set(), "Backward")
        # backward move from empty selection is not possible no scores are computed
        self.assertEqual(0, len(scores))

        selection = {d["LSTAT"], d["RM"], d["CHAS"]}
        scores = sc.compute_feature_scores(self.housing, selection, set(), "Backward")
        self.assertEqual(3, len(scores))
        self.assertAlmostEqual(-0.086, scores[d["RM"]], places=1)
        self.assertAlmostEqual(-0.011, scores[d["CHAS"]], places=1)
        self.assertAlmostEqual(-0.158, scores[d["LSTAT"]], places=1)

        selected = {d["LSTAT"], d["RM"]}
        scores = sc.compute_feature_scores(self.housing, selected, set(), "Backward")
        self.assertEqual(2, len(scores))
        self.assertAlmostEqual(-0.091, scores[d["RM"]], places=1)
        self.assertAlmostEqual(-0.157, scores[d["LSTAT"]], places=1)

    def test_get_methods(self):
        methods = Scoring.get_methods((ContinuousVariable, DiscreteVariable))
        self.assertListEqual(
            ["R2", "RMSE", "MAE", "AUC", "CA", "F1", "Prec", "Recall"], methods
        )
        methods = Scoring.get_methods((DiscreteVariable,))
        self.assertListEqual(["AUC", "CA", "F1", "Prec", "Recall"], methods)
        methods = Scoring.get_methods((ContinuousVariable,))
        self.assertListEqual(["R2", "RMSE", "MAE"], methods)


class TestStopping(TestCase):
    def setUp(self):
        self.iris = Table("iris")
        self.housing = Table("housing")

    def test_n_features(self):
        rule = ("N-features", {"n_features": 5})
        st = Stopping(rule, "Forward", self.housing, Mock())
        self.assertFalse(st.finished(set(), {}))
        self.assertFalse(st.finished(set(self.housing.domain.attributes[:4]), {}))
        self.assertTrue(st.finished(set(self.housing.domain.attributes[:5]), {}))
        self.assertTrue(st.finished(set(self.housing.domain.attributes[:6]), {}))

    def test_best_socre(self):
        d = self.housing.domain
        rule = ("Score delta", {"threshold": 1.0})
        st = Stopping(rule, "Forward", self.housing, Mock())
        self.assertFalse(st.finished(set(), {d["CRIM"]: 1.0, d["DIS"]: 0.1}))
        self.assertTrue(st.finished(set(), {d["CRIM"]: 0.99, d["DIS"]: 0.1}))

    def test_bic(self):
        pass  # todo

    def test_aic(self):
        pass  # todo


class TestStepwiseFeatureSelection(TestCase):
    def setUp(self):
        self.iris = Table("iris")
        self.housing = Table("housing")

        evaluation = ("Random split", {"test_size": 0.2})
        self.sfs = StepwiseFeatureSelection("Forward", "CA", evaluation, None)
        self.sfs.set_data_and_scoring(self.iris, set(), set(), "CA")

    def assert_scores_almost_equal(self, expected, results):
        self.assertEqual(len(expected), len(results))
        for var, score in expected.items():
            self.assertAlmostEqual(results[var], score, delta=0.1)

    def test_compute_scores_CA_forward(self):
        params = {"Cross validation": {"k": 3}}
        for v in Scoring.VALIDATION_METHODS:
            domain = self.iris.domain
            sfs = StepwiseFeatureSelection(
                "Forward", "CA", (v, params.get(v, {})), LogisticRegressionLearner()
            )
            sfs.set_data_and_scoring(self.iris, set(), set(), "CA")
            expected_ = {
                domain["sepal length"]: 0.426,
                domain["sepal width"]: 0.23,
                domain["petal length"]: 0.62,
                domain["petal width"]: 0.63,
            }
            self.assert_scores_almost_equal(expected_, sfs.scores)

            sfs = StepwiseFeatureSelection(
                "Forward", "CA", (v, params.get(v, {})), LogisticRegressionLearner()
            )
            sfs.set_data_and_scoring(self.iris, {domain["sepal width"]}, set(), "CA")

            expected_ = {
                domain["sepal length"]: 0.25,
                domain["petal length"]: 0.39,
                domain["petal width"]: 0.38,
            }
            self.assert_scores_almost_equal(expected_, sfs.scores)

    def test_compute_scores_CA_backward(self):
        params = {"Cross validation": {"k": 3}}
        for v in Scoring.VALIDATION_METHODS:
            domain = self.iris.domain
            all_vars = set(domain.attributes)
            sfs = StepwiseFeatureSelection(
                "Backward", "CA", (v, params.get(v, {})), LogisticRegressionLearner()
            )
            sfs.set_data_and_scoring(self.iris, all_vars, set(), "CA")
            expected_ = {
                domain["sepal length"]: 0.0,
                domain["sepal width"]: 0.0,
                domain["petal length"]: -0.01,
                domain["petal width"]: -0.01,
            }
            self.assert_scores_almost_equal(expected_, sfs.scores)

            vars_ = all_vars - {domain["petal width"]}
            sfs.set_data_and_scoring(self.iris, vars_, set(), "CA")
            expected_ = {
                domain["sepal length"]: 0.0,
                domain["sepal width"]: 0.0,
                domain["petal length"]: -0.13,
            }

            self.assert_scores_almost_equal(expected_, sfs.scores)

    def test_step_forward(self):
        sfs = StepwiseFeatureSelection(
            "Forward", "CA", ("Cross validation", {"k": 5}), LogisticRegressionLearner()
        )
        sfs.set_data_and_scoring(self.iris, set(), set(), "CA")

        self.assertSetEqual(set(), sfs.selected)
        expected_sc = set(self.iris.domain.attributes)
        self.assertEqual(expected_sc, set(sfs.scores.keys()))

        sfs.step()
        expected_selected = {self.iris.domain["petal width"]}
        self.assertSetEqual(expected_selected, sfs.selected)
        self.assertSetEqual(expected_sc - expected_selected, set(sfs.scores.keys()))

        sfs.step()
        expected_selected.add(self.iris.domain["petal length"])
        self.assertSetEqual(expected_selected, sfs.selected)
        self.assertSetEqual(expected_sc - expected_selected, set(sfs.scores.keys()))

        sfs.step()
        expected_selected.add(self.iris.domain["sepal length"])
        self.assertSetEqual(expected_selected, sfs.selected)
        self.assertSetEqual(expected_sc - expected_selected, set(sfs.scores.keys()))

        sfs.step()
        expected_selected.add(self.iris.domain["sepal width"])
        self.assertSetEqual(expected_selected, sfs.selected)
        self.assertSetEqual(expected_sc - expected_selected, set(sfs.scores.keys()))

        sfs.step()
        self.assertSetEqual(expected_selected, sfs.selected)
        self.assertSetEqual(expected_sc - expected_selected, set(sfs.scores.keys()))

    def test_step_backward(self):
        expected_sc = set(self.iris.domain.attributes)
        lr = LogisticRegressionLearner(random_state=0)
        ev = ("Cross validation", {"k": 5})
        sfs = StepwiseFeatureSelection("Backward", "CA", ev, lr)
        sfs.set_data_and_scoring(self.iris, expected_sc, set(), "CA")

        self.assertSetEqual(expected_sc, sfs.selected)
        self.assertEqual(expected_sc, set(sfs.scores.keys()))

        sfs.step()
        expected_sc.remove(self.iris.domain["sepal width"])
        self.assertSetEqual(expected_sc, sfs.selected)
        self.assertSetEqual(expected_sc, set(sfs.scores.keys()))

        sfs.step()
        expected_sc.remove(self.iris.domain["sepal length"])
        self.assertSetEqual(expected_sc, sfs.selected)
        self.assertSetEqual(expected_sc, set(sfs.scores.keys()))

        sfs.step()
        expected_sc.remove(self.iris.domain["petal length"])
        self.assertSetEqual(expected_sc, sfs.selected)
        self.assertSetEqual(expected_sc, set(sfs.scores.keys()))

        sfs.step()
        expected_sc.remove(self.iris.domain["petal width"])
        self.assertSetEqual(expected_sc, sfs.selected)
        self.assertSetEqual(expected_sc, set(sfs.scores.keys()))

        sfs.step()
        self.assertSetEqual(expected_sc, sfs.selected)
        self.assertSetEqual(expected_sc, set(sfs.scores.keys()))

    def test_set_learner(self):
        sfs = StepwiseFeatureSelection(
            "Forward", "CA", ("Cross validation", {"k": 5}), LinearRegressionLearner()
        )
        # set learner when no data
        sfs.set_learner(LogisticRegressionLearner())
        self.assertDictEqual({}, sfs.scores)

        # set learner with data
        sfs.set_data_and_scoring(self.iris, set(), set(), "CA")
        scores_before = sfs.scores.copy()
        sfs.set_learner(RandomForestLearner())
        # test that scores recomputed
        self.assertNotEqual(scores_before, sfs.scores)

    def test_compute_scores(self):
        # scores with constant learner
        scores = self.sfs.compute_scores()
        ex = {"AUC": 0.5, "CA": 0.33, "F1": 0.167, "Prec": 0.111, "Recall": 0.333}
        self.assert_scores_almost_equal(ex, scores)

        # scores logistic regression (on best feature)
        self.sfs.step()
        scores = self.sfs.compute_scores()
        ex = {"AUC": 0.99, "CA": 0.94, "F1": 0.94, "Prec": 0.94, "Recall": 0.94}
        self.assert_scores_almost_equal(ex, scores)

        # regression case
        evaluation = ("Random split", {"test_size": 0.2})
        sfs = StepwiseFeatureSelection("Forward", "R2", evaluation, None)
        sfs.set_data_and_scoring(self.housing, set(), set(), "R2")

        # scores with constant learner
        scores = sfs.compute_scores()
        self.assert_scores_almost_equal({"R2": 0.00, "RMSE": 9.03, "MAE": 6.25}, scores)

        # scores logistic regression (on best feature)
        sfs.step()
        scores = sfs.compute_scores()
        self.assert_scores_almost_equal({"R2": 0.43, "RMSE": 6.81, "MAE": 4.86}, scores)

    def test_include(self):
        d = self.iris.domain
        self.sfs.include({d["petal length"]})
        self.assertSetEqual({d["petal length"]}, self.sfs.selected)
        self.sfs.include({d["petal width"]})
        self.assertSetEqual({d["petal width"], d["petal length"]}, self.sfs.selected)
        self.assertSetEqual({d["sepal length"], d["sepal width"]}, set(self.sfs.scores))
        # wrong feature
        with self.assertRaises(AssertionError):
            self.sfs.include({self.housing.domain["AGE"]})

    def test_exclude(self):
        d = self.iris.domain
        self.sfs.set_data_and_scoring(self.iris, set(d.attributes), set(), "CA")

        self.sfs.exclude({d["petal length"]})
        exp = {d["petal width"], d["sepal width"], d["sepal length"]}
        self.assertSetEqual(exp, self.sfs.selected)
        self.sfs.exclude({d["petal width"]})
        self.assertSetEqual({d["sepal width"], d["sepal length"]}, self.sfs.selected)
        self.assertSetEqual({d["petal length"], d["petal width"]}, set(self.sfs.scores))

    def test_lock(self):
        d = self.iris.domain

        # lock not selected feature, should not be selected
        self.sfs.lock({d["petal length"]})
        self.assertSetEqual({d["petal length"]}, self.sfs.locked)
        for _ in range(4):
            self.sfs.step()
        self.assertNotIn(d["petal length"], self.sfs.selected)

        # lock selected feature and try to step backward, should stay selected
        self.sfs.lock({d["sepal width"]})
        self.assertSetEqual({d["petal length"], d["sepal width"]}, self.sfs.locked)
        self.sfs.set_direction("Backward")
        for _ in range(4):
            self.sfs.step()
        self.assertIn(d["sepal width"], self.sfs.selected)

    def test_unlock(self):
        d = self.iris.domain
        locked = {d["petal length"], d["sepal width"]}
        self.sfs.set_data_and_scoring(self.iris, {d["petal length"]}, locked, "CA")

        self.assertSetEqual(locked, self.sfs.locked)
        # unlock non-selected feature try to select it
        self.assertNotIn(d["sepal width"], self.sfs.selected)
        self.sfs.unlock({d["sepal width"]})
        for _ in range(4):
            self.sfs.step()
        self.assertIn(d["sepal width"], self.sfs.selected)

        # unlock selected feature and try to remove it from selection
        self.sfs.set_direction("Backward")
        self.assertIn(d["petal length"], self.sfs.selected)
        self.sfs.unlock({d["petal length"]})
        for _ in range(4):
            self.sfs.step()
        self.assertNotIn(d["petal length"], self.sfs.selected)

    def test_set_direction(self):
        self.assertEqual("Forward", self.sfs.direction)
        for _ in range(4):
            self.sfs.step()
        self.assertSetEqual(set(self.iris.domain.attributes), self.sfs.selected)

        self.sfs.set_direction("Backward")
        self.assertEqual("Backward", self.sfs.direction)
        for _ in range(4):
            self.sfs.step()
        self.assertSetEqual(set(), self.sfs.selected)

        self.sfs.set_direction("Forward")
        self.assertEqual("Forward", self.sfs.direction)
        for _ in range(4):
            self.sfs.step()
        self.assertSetEqual(set(self.iris.domain.attributes), self.sfs.selected)

    def test_set_score(self):
        self.assertEqual("CA", self.sfs.scorer.method)
        curr_scores = self.sfs.scores

        self.sfs.set_score("AUC")
        self.assertEqual("AUC", self.sfs.scorer.method)
        self.assertNotEqual(curr_scores, self.sfs.scores)

    def test_set_validation(self):
        evaluation = ("Random split", {"test_size": 0.2})
        self.assertEqual(evaluation, self.sfs.scorer.validation)
        curr_scores = self.sfs.scores

        evaluation = ("Cross validation", {"k": 3})
        self.sfs.set_validation(evaluation)
        self.assertEqual(evaluation, self.sfs.scorer.validation)
        self.assertNotEqual(curr_scores, self.sfs.scores)

    def test_history(self):
        d = self.iris.domain
        # step added to history
        self.sfs.step()
        expected = [("selected", "in", {d["petal length"]})]
        self.assertListEqual(expected, self.sfs.history)

        # step backward added to history
        self.sfs.set_direction("Backward")
        self.sfs.step()
        expected.append(("selected", "out", {d["petal length"]}))
        self.assertListEqual(expected, self.sfs.history)

        # include added to history
        self.sfs.include({d["sepal length"]})
        expected.append(("selected", "in", {d["sepal length"]}))
        self.assertListEqual(expected, self.sfs.history)

        # test exclude added to history
        self.sfs.exclude({d["sepal length"]})
        expected.append(("selected", "out", {d["sepal length"]}))
        self.assertListEqual(expected, self.sfs.history)

        # test lock added to history
        self.sfs.lock({d["sepal width"]})
        expected.append(("locked", "in", {d["sepal width"]}))
        self.assertListEqual(expected, self.sfs.history)

        # test step back removed from history
        self.sfs.unlock({d["sepal width"]})
        expected.append(("locked", "out", {d["sepal width"]}))
        self.assertListEqual(expected, self.sfs.history)

    def test_step_back(self):
        d = self.iris.domain
        self.sfs.history = [
            ("selected", "in", {d["sepal length"]}),
            ("selected", "out", {d["sepal length"]}),
            ("locked", "in", {d["sepal width"]}),
            ("locked", "out", {d["sepal width"]}),
        ]
        self.assertSetEqual(set(), self.sfs.locked)
        self.assertSetEqual(set(), self.sfs.selected)
        self.assertSetEqual(set(d.attributes), set(self.sfs.scores))

        self.sfs.step_back()
        self.assertSetEqual({d["sepal width"]}, self.sfs.locked)
        self.assertSetEqual(set(), self.sfs.selected)
        self.assertSetEqual(set(d.attributes), set(self.sfs.scores))

        self.sfs.step_back()
        self.assertSetEqual(set(), self.sfs.locked)
        self.assertSetEqual(set(), self.sfs.selected)
        self.assertSetEqual(set(d.attributes), set(self.sfs.scores))

        self.sfs.step_back()
        self.assertSetEqual(set(), self.sfs.locked)
        self.assertSetEqual({d["sepal length"]}, self.sfs.selected)
        self.assertSetEqual(set(d.attributes[1:]), set(self.sfs.scores))

        self.sfs.step_back()
        self.assertSetEqual(set(), self.sfs.locked)
        self.assertSetEqual(set(), self.sfs.selected)
        self.assertSetEqual(set(d.attributes), set(self.sfs.scores))

    def test_run_n_features(self):
        evaluation = ("Random split", {"test_size": 0.1})
        sfs = StepwiseFeatureSelection("Forward", "R2", evaluation, None)
        sfs.set_data_and_scoring(self.housing, set(), set(), "R2")

        d = self.housing.domain
        sfs.run(("N-features", {"n_features": 3}))
        exp = {d["LSTAT"], d["DIS"], d["ZN"]}
        self.assertSetEqual(exp, sfs.selected)

        sfs.run(("N-features", {"n_features": 4}))
        exp = {d["LSTAT"], d["DIS"], d["ZN"], d["CHAS"]}
        self.assertSetEqual(exp, sfs.selected)

    def test_run_score_delta(self):
        evaluation = ("Random split", {"test_size": 0.1})
        sfs = StepwiseFeatureSelection("Forward", "R2", evaluation, None)
        sfs.set_data_and_scoring(self.housing, set(), set(), "R2")

        d = self.housing.domain
        sfs.run(("Score delta", {"threshold": 0.02}))
        exp = {d["LSTAT"], d["DIS"], d["ZN"]}
        self.assertSetEqual(exp, sfs.selected)

        sfs.run(("Score delta", {"threshold": 0.015}))
        exp = {d["LSTAT"], d["DIS"], d["ZN"], d["CHAS"], d["NOX"]}
        self.assertSetEqual(exp, sfs.selected)

    def test_run_bic(self):
        evaluation = ("Random split", {"test_size": 0.1})
        sfs = StepwiseFeatureSelection("Forward", "R2", evaluation, None)
        sfs.set_data_and_scoring(self.housing, set(), set(), "R2")

        d = self.housing.domain
        sfs.run(("Minimum BIC", {}))
        exp = {d["B"], d["CHAS"], d["DIS"], d["LSTAT"], d["NOX"], d["ZN"]}
        self.assertSetEqual(exp, sfs.selected)

    def test_run_aic(self):
        evaluation = ("Random split", {"test_size": 0.1})
        sfs = StepwiseFeatureSelection("Forward", "R2", evaluation, None)
        sfs.set_data_and_scoring(self.housing, set(), set(), "R2")

        d = self.housing.domain
        sfs.run(("Minimum AICc", {}))
        exp = {d["B"], d["CHAS"], d["DIS"], d["LSTAT"], d["NOX"], d["ZN"]}
        self.assertSetEqual(exp, sfs.selected)

    def test_run_to_end(self):
        """Test run for cases when criteria not met. Stop when no more features"""
        d = self.iris.domain
        all_feat = set(self.iris.domain.attributes)
        evaluation = ("Random split", {"test_size": 0.1})
        sfs = StepwiseFeatureSelection("Forward", "AUC", evaluation, None)
        sfs.set_data_and_scoring(self.iris, set(), set(), "AUC")
        sfs.run(("N-features", {"n_features": 10}))
        self.assertSetEqual(all_feat, sfs.selected)

        sfs = StepwiseFeatureSelection("Backward", "AUC", evaluation, None)
        sfs.set_data_and_scoring(self.iris, all_feat, set(), "AUC")
        sfs.run(("N-features", {"n_features": -1}))
        self.assertSetEqual(set(), sfs.selected)

        # try also with locked features
        locked = {d["petal length"]}
        sfs = StepwiseFeatureSelection("Forward", "AUC", evaluation, None)
        sfs.set_data_and_scoring(self.iris, set(), {d["petal length"]}, "AUC")
        sfs.run(("N-features", {"n_features": 10}))
        self.assertSetEqual(all_feat - locked, sfs.selected)

        locked = {d["petal length"]}
        sfs = StepwiseFeatureSelection("Backward", "AUC", evaluation, None)
        sfs.set_data_and_scoring(self.iris, all_feat, {d["petal length"]}, "AUC")
        sfs.run(("N-features", {"n_features": -1}))
        self.assertSetEqual(locked, sfs.selected)


class TestFeatureSelectionPreprocessor(TestCase):
    def setUp(self):
        self.iris = Table("iris")
        self.housing = Table("housing")
        self.lr = LinearRegressionLearner()
        self.lg = LogisticRegressionLearner()
        self.validation = ("Random split", {"test_size": 0.3})

    def test_forward(self):
        st = ("N-features", {"n_features": 3})
        pp = FeatureSelectionPreprocessor("Forward", "R2", self.validation, st, self.lr)
        data = pp(self.housing)
        self.assertEqual(len(self.housing), len(data))
        self.assertSetEqual({"LSTAT", "RM", "B"}, ft_set(data.domain.attributes))

        st = ("Score delta", {"threshold": 0.02})
        pp = FeatureSelectionPreprocessor("Forward", "R2", self.validation, st, self.lr)
        data = pp(self.housing)
        self.assertEqual(len(self.housing), len(data))
        self.assertSetEqual({"LSTAT", "RM", "B", "RM"}, ft_set(data.domain.attributes))

    def test_backward(self):
        st = ("N-features", {"n_features": 4})
        p = FeatureSelectionPreprocessor("Backward", "R2", self.validation, st, self.lr)
        data = p(self.housing)
        self.assertEqual(len(self.housing), len(data))
        self.assertSetEqual(
            {"LSTAT", "RM", "CHAS", "B"}, ft_set(data.domain.attributes)
        )

        st = ("Score delta", {"threshold": -0.02})
        p = FeatureSelectionPreprocessor("Backward", "R2", self.validation, st, self.lr)
        data = p(self.housing)
        self.assertEqual(len(self.housing), len(data))
        self.assertSetEqual({"LSTAT", "RM", "B"}, ft_set(data.domain.attributes))

    def test_incompatible_data_learner(self):
        st = ("N-features", {"n_features": 4})
        pp = FeatureSelectionPreprocessor("Forward", "R2", self.validation, st, self.lr)
        with self.assertRaises(ValueError) as err:
            pp(self.iris)
        self.assertEqual(
            "Linear regression doesn't support categorical class variable. "
            "Please use different learner instead",
            str(err.exception),
        )

        pp = FeatureSelectionPreprocessor("Forward", "R2", self.validation, st, self.lg)
        with self.assertRaises(ValueError) as err:
            pp(self.housing)
        self.assertEqual(
            "Logistic regression doesn't support numeric class variable. "
            "Please use different learner instead",
            str(err.exception),
        )

    def test_incompatible_data_score(self):
        st = ("N-features", {"n_features": 4})
        for score in Scoring.SCORING_METHODS[ContinuousVariable]:
            pp = FeatureSelectionPreprocessor(
                "Forward", score, self.validation, st, self.lg
            )
            with self.assertRaises(ValueError) as err:
                pp(self.iris)
            self.assertEqual(
                f"{score} score doesn't work with categorical class variable. "
                "Please use different scoring method instead",
                str(err.exception),
            )

        for score in Scoring.SCORING_METHODS[DiscreteVariable]:
            pp = FeatureSelectionPreprocessor(
                "Forward", score, self.validation, st, self.lr
            )
            with self.assertRaises(ValueError) as err:
                pp(self.housing)
            self.assertEqual(
                f"{score} score doesn't work with numeric class variable. "
                "Please use different scoring method instead",
                str(err.exception),
            )


if __name__ == "__main__":
    unittest.main()
