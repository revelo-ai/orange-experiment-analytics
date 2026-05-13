import time
from functools import partial
from numbers import Number
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from Orange.base import Learner, Model
from Orange.classification import LogisticRegressionLearner
from Orange.classification.base_classification import LearnerClassification
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table, Variable
from Orange.data.variable import VariableMeta
from Orange.evaluation import (
    AUC,
    CA,
    F1,
    MAE,
    MSE,
    R2,
    RMSE,
    CrossValidation,
    LogLoss,
    Precision,
    Recall,
    Results,
    ShuffleSplit,
    TestOnTrainingData,
)
from Orange.evaluation.scoring import Score
from Orange.evaluation.testing import Validation
from Orange.modelling import ConstantLearner
from Orange.preprocess import Preprocess
from Orange.regression import LinearRegressionLearner
from Orange.regression.base_regression import LearnerRegression
from Orange.util import dummy_callback, wrap_callback
from Orange.widgets.utils.concurrent import TaskState


class Constant0Model(Model):
    """
    Model that always predicts 0. Linear regression with no attributes and no intercept.
    """

    def __call__(self, data: Table, **_) -> np.ndarray:
        return np.zeros(len(data))


class Constant0Learner(Learner):
    """Learner that returns model that always predicts 0."""

    def __call__(self, _, **__) -> Constant0Model:
        return Constant0Model()


def _filter_data(data: Table, selected_features: Set[Variable]) -> Table:
    """Creates tables with only attributes in selected_features"""
    domain = Domain(
        [a for a in data.domain.attributes if a in selected_features],
        class_vars=data.domain.class_vars,
    )
    return data.transform(domain)


def select_learner(data: Table, learner: Learner) -> Learner:
    """Select learner based on data and selected learner. See comments for details."""
    if len(data.domain.attributes) == 0:
        if isinstance(learner, LinearRegressionLearner) and not learner.fit_intercept:
            # when are without attributes and intercept is not used it is not correct
            # to use constant learner since it would mean fitting intercept
            # use model that predict 0 instead
            # turning off intercept is only supported for linear regression
            learner = Constant0Learner()
        else:
            # in all other cases (when no attributes) intercept is used or model do not
            # have intercept-like parameter, use constant
            learner = ConstantLearner()
    return learner


def _predict(data: Table, learner: Learner, method: Validation) -> Results:
    """Run validation on data with learner using learner returned by select_learner"""
    return method(data, [select_learner(data, learner)])


def supported_types(learner: Learner) -> Tuple[VariableMeta]:
    """Returns tuple with variables types that the learner supports"""
    if isinstance(learner, LearnerClassification):
        types = (DiscreteVariable,)
    elif isinstance(learner, LearnerRegression):
        types = (ContinuousVariable,)
    else:  # else support both types
        types = (ContinuousVariable, DiscreteVariable)
    return types


class Scoring:
    REGRESSION_METHODS = [R2, RMSE, MAE]
    CLASSIFICATION_METHODS = [AUC, CA, F1, Precision, Recall]
    SCORING_METHODS = {
        ContinuousVariable: {m.name: m for m in REGRESSION_METHODS},
        DiscreteVariable: {m.name: m for m in CLASSIFICATION_METHODS},
    }
    ALL_SCORING_METHODS = {k: v for d in SCORING_METHODS.values() for k, v in d.items()}
    VALIDATION_METHODS = {
        "Cross validation": CrossValidation,
        # "Test on training data": TestOnTrainingData,  todo: decide later
        "Random split": partial(ShuffleSplit, n_resamples=1),
    }
    DEFAULT_LEARNER = {
        ContinuousVariable: LinearRegressionLearner,
        DiscreteVariable: partial(LogisticRegressionLearner, random_state=0),
    }

    def __init__(
        self,
        scoring_method: str,
        validation: Tuple[str, Dict[str, Number]],
        learner: Learner = None,
    ):
        self.learner = learner
        self.method = scoring_method
        self.validation = validation
        self.baseline_evaluation = None

    def get_learner(self, data: Table) -> Learner:
        """
        Return either user-selected learner or default learner for data if user
        doesn't provide one
        """
        return self.learner or self.DEFAULT_LEARNER[type(data.domain.class_var)]()

    def __evaluate(self, data: Table) -> Results:
        """Run evaluation on the data"""
        validation = self.VALIDATION_METHODS[self.validation[0]]
        kwargs = {"random_state": 0} if validation != TestOnTrainingData else {}
        method = validation(**kwargs, **self.validation[1])
        return _predict(data, self.get_learner(data), method)

    @staticmethod
    def __score(ev_results: Results, method: Score) -> float:
        """Compute scores on the evaluation result with provided scoring method"""
        kwargs = {"average": "weighted"} if method in (F1, Precision, Recall) else {}
        # transform from np.float64 to float which QSortFilterProxyModel can compare
        return float(method(ev_results, **kwargs)[0])

    def __check_data(self, data: Table):
        """
        Check if data compatible with scoring method and learner.
        It is specially important for preprocessor.
        """
        # check class in data
        if not data.domain.class_vars:
            raise ValueError("Data input requires a target variable.")

        # check data and learner compatible
        learner = self.get_learner(data)
        var = data.domain.class_var
        v_type = "categorical" if isinstance(var, DiscreteVariable) else "numeric"
        if not isinstance(var, supported_types(learner)):
            raise ValueError(
                f"{learner.name.capitalize()} doesn't support {v_type} class variable. "
                "Please use different learner instead"
            )

        # check data and score compatible
        if self.method not in self.SCORING_METHODS[type(var)]:
            raise ValueError(
                f"{self.method} score doesn't work with {v_type} class variable. "
                "Please use different scoring method instead"
            )

    def score_selection(
        self, data: Table, selected_features: Set[Variable], is_baseline: bool = False
    ) -> float:
        """Score selected features with selected scoring method and learner"""
        self.__check_data(data)
        data = _filter_data(data, selected_features)
        res = self.__evaluate(data)
        score = self.__score(res, self.ALL_SCORING_METHODS[self.method])
        if is_baseline:
            self.baseline_evaluation = res
        return score

    def score_selection_all_methods(self, data: Optional[Table]) -> Dict[str, float]:
        """Score data with all methods for certain data class type"""
        results = {}
        if data is not None:
            self.__check_data(data)
            methods = self.SCORING_METHODS[type(data.domain.class_var)].items()
            for m_name, method in methods:
                if self.baseline_evaluation:
                    results[m_name] = self.__score(self.baseline_evaluation, method)
                else:
                    results[m_name] = "NA"
        return results

    def compute_feature_scores(
        self,
        data: Table,
        selected: Set[Variable],
        locked: Set[Variable],
        direction: str,
        callback: Callable = dummy_callback,
    ) -> Dict[Variable, float]:
        """Compute score delta for all features"""
        scores = {}
        baseline = self.score_selection(data, selected, True)

        def prepare_features(fe):
            return selected | {fe} if direction == "Forward" else selected - {fe}

        if direction == "Forward":
            candidates = set(data.domain.attributes) - selected - locked
        else:
            candidates = selected - locked
        for i, f in enumerate(candidates):
            callback((i + 1) / len(candidates))
            s = self.score_selection(data, prepare_features(f))
            # MSE and MAE need to be minimized for improvement other ar maximized
            scores[f] = baseline - s if self.method in ("RMSE", "MAE") else s - baseline
        return scores

    @staticmethod
    def get_methods(types: Iterable) -> List[str]:
        """Get list of scoring methods for class variable type"""
        return [m for type_ in types for m in Scoring.SCORING_METHODS[type_]]


class Stopping:
    def __init__(
        self,
        rule: Tuple[str, Dict],
        direction: str,
        data: Table,
        learner: Learner,
    ):
        fun = self.RULES[rule[0]]
        self.fun = partial(fun, self, **rule[1])
        self.direction = direction
        self.data = data
        self.learner = learner

    def __n_features(self, selected_f: Set[Variable], _, n_features: int = 10) -> bool:
        """Method that check if number-features stopping rule is met."""
        if self.direction == "Forward":
            return len(selected_f) >= n_features
        else:
            return len(selected_f) <= n_features

    def __best_score(
        self, _, scores: Dict[Variable, float], threshold: float = 1
    ) -> bool:
        """
        Method that check if best score rule is met. Best score rule is met when
        there is no feature that can further improve the model.
        """
        return all(v < threshold for v in scores.values())

    def __bic(self, predictions: Results, k: int) -> float:
        """Computes BIC score on evaluation results"""
        n = len(self.data)
        if isinstance(self.data.domain.class_var, ContinuousVariable):
            # Bic derivation for linear regression (similar than AIC)
            # https://en.wikipedia.org/wiki/Bayesian_information_criterion
            ll = MSE()(predictions)[0]
            return n * np.log(ll) + k * np.log(n)
        else:
            # Basic BIC formula. I didn't find a special derivation for logistic
            # regression (similar to one for linear regression) so we will use basic
            # formula for logistic regression and other classification models.
            ll = LogLoss()(predictions, normalize=False)[0]
            return 2 * np.log(ll) + k * np.log(n)

    def __aicc(self, predictions: Results, k: int) -> float:
        """Computes AICc score on evaluation results"""
        n = len(self.data)
        if isinstance(self.data.domain.class_var, ContinuousVariable):
            # Derivation of AICc for linear regression. Derivations:
            # https://en.wikipedia.org/wiki/Akaike_information_criterion#Comparison_with_least_squares
            # https://stats.stackexchange.com/questions/261273/how-can-i-apply-akaike-information-criterion-and-calculate-it-for-linear-regress
            # https://stats.stackexchange.com/questions/583479/why-is-aic-computed-with-a-term-containing-logssr-instead-of-ssr
            ll = MSE()(predictions)[0]
            return 2 * k + n * np.log(ll) + (2 * k**2 + 2 * k) / (n - k - 1)
        else:
            # Basic AIC formula. I didn't find a special derivation for logistic
            # regression (similar to one for linear regression) so we will use basic
            # formula for logistic regression and other classification models.
            ll = LogLoss()(predictions, normalize=False)[0]
            return 2 * k + 2 * np.log(ll) + (2 * k**2 + 2 * k) / (n - k - 1)

    def __compute_score(self, data: Table, score_fun: Callable) -> float:
        """Compute BIC or AICc score"""
        # fit models and make predictions
        actual_learner = select_learner(data, self.learner)
        predictions = TestOnTrainingData()(data, [actual_learner])

        # get appropriate likelihood function and compute likelihoods
        has_intercept = getattr(actual_learner, "fit_intercept", False)
        k = len(data.domain.attributes) + (1 if has_intercept else 0)
        return score_fun(self, predictions, k)

    def __score_diff(
        self,
        selected_features: Set[Variable],
        scores: Dict[Variable, float],
        score_fun: Callable,
        **_,
    ) -> bool:
        """Prepare data and check if AICc or BIC lowers with next feature added"""
        # find feature will be added/removed to/from the model next
        best_feature = max(scores, key=scores.get)
        if self.direction == "Forward":
            new_features = selected_features | {best_feature}
        else:
            new_features = selected_features - {best_feature}

        # prepare data with adding/removing parameters
        data_current = _filter_data(self.data, selected_features)
        data_new = _filter_data(self.data, new_features)
        sc_current = self.__compute_score(data_current, score_fun)
        sc_new = self.__compute_score(data_new, score_fun)
        return sc_current <= sc_new

    RULES = {
        "N-features": __n_features,
        "Score delta": __best_score,
        "Minimum AICc": partial(__score_diff, score_fun=__aicc),
        "Minimum BIC": partial(__score_diff, score_fun=__bic),
    }

    def finished(self, selected_features: Set[Variable], scores: Dict) -> bool:
        """
        Test if feature adding is finished using certain stopping rule.

        Parameters
        ----------
        selected_features
            Current feature selection
        scores
            Current scores all features that should be added/excluded

        Returns
        -------
        Boolean indicating whether feature adding should be finished
        """
        return self.fun(selected_features, scores)


class StepwiseFeatureSelection:
    DIRECTIONS = ["Forward", "Backward"]

    def __init__(
        self,
        direction: str,
        scoring_method: str,
        validation: Tuple[str, Dict[str, Number]],
        learner: Learner = None,
    ):
        self.data = None
        self.scorer = Scoring(scoring_method, validation, learner)
        self.selected = set()
        self.locked = set()
        self.scores = {}
        self.direction = direction
        self.history = []
        self.stop = False

    def set_data(self, data: Optional[Table], selected: Set[Variable]):
        assert all(feature in data.domain.attributes for feature in selected)
        self.data = data
        self.selected = selected.copy()

    def set_data_and_scoring(
        self,
        data: Table,
        selected: Set[Variable],
        locked: Set[Variable],
        scoring_method: str,
        callback: Callable = dummy_callback,
    ):
        """
        Set data, scoring method and selected/locked features. It is all done
        with one setter since widget changes all those parameters on new data.
        """
        assert all(feature in data.domain.attributes for feature in locked)
        self.set_data(data, selected)
        self.locked = locked.copy()
        self.scores = {}
        self.scorer.method = scoring_method
        self.compute_feature_scores(callback)
        self.history = []

    def set_learner(self, learner: Learner, callback: Callable = dummy_callback):
        """Set learner used for scoring"""
        self.scorer.learner = learner
        self.compute_feature_scores(callback)

    def compute_feature_scores(self, callback: Callable = dummy_callback):
        """Compute scores deltas when scoring, data or selected features change"""
        if self.data is not None:
            self.scores = self.scorer.compute_feature_scores(
                self.data, self.selected, self.locked, self.direction, callback
            )
        else:
            self.scores = {}

    def compute_scores(self) -> Dict[str, float]:
        """Compute all scores for widget scores table"""
        return self.scorer.score_selection_all_methods(self.data)

    def include(self, attributes: Set[Variable], task_state: Optional[TaskState] = None):
        """Include features. Needed for user's manual actions"""
        def callback(progress: float):
            if task_state:
                task_state.set_progress_value(progress * 100)

        assert attributes <= set(self.data.domain.attributes)
        assert not (attributes & self.locked)
        self.selected |= attributes
        if task_state:
            task_state.set_partial_result("entered")
        self.add_to_history("selected", "in", attributes)
        self.compute_feature_scores(callback)

    def exclude(self, attributes: Set[Variable], task_state: Optional[TaskState] = None):
        """Exclude features. Needed for user's manual actions"""
        def callback(progress: float):
            if task_state:
                task_state.set_progress_value(progress * 100)

        assert not (attributes & self.locked)
        self.selected -= attributes
        if task_state:
            task_state.set_partial_result("entered")
        self.add_to_history("selected", "out", attributes)
        self.compute_feature_scores(callback)

    def lock(self, attributes: Set[Variable]):
        """Lock features. Needed for user's manual actions"""
        assert attributes <= set(self.data.domain.attributes)
        self.locked |= attributes
        self.add_to_history("locked", "in", attributes)

        for attr in attributes:
            if attr in self.scores:
                del self.scores[attr]

    def unlock(self, attributes: Set[Variable]):
        """Unlock features. Needed for user's manual actions"""
        self.locked -= attributes
        self.add_to_history("locked", "out", attributes)

        if self.data is not None:
            domain = Domain(list(attributes), self.data.domain.class_vars)
            data = self.data.transform(domain)
            scores = self.scorer.compute_feature_scores(
                data, self.selected, self.locked, self.direction, dummy_callback
            )
            self.scores.update(scores)

    def set_direction(self, direction: str, callback: Callable = dummy_callback):
        """Set direction of feature selection"""
        assert direction in self.DIRECTIONS, f"Direction {direction} not supported"
        self.direction = direction
        self.compute_feature_scores(callback)

    def set_score(self, scoring_method: str, callback: Callable = dummy_callback):
        """Set model scoring method"""
        self.scorer.method = scoring_method
        self.compute_feature_scores(callback)

    def set_validation(
        self, validation: Tuple[str, Dict], callback: Callable = dummy_callback
    ):
        """Set validation (evaluation) type. E.g. cross-validation"""
        self.scorer.validation = validation
        self.compute_feature_scores(callback)

    def step(self, task_state: Optional[TaskState] = None):
        """Make on step in selected direction."""
        # if not all non-locked feature included/excluded select best feature
        scores = {f: s for f, s in self.scores.items() if f not in self.locked}
        if scores:
            best_feature = max(scores, key=scores.get)
            # include/exclude also recompute scores
            if self.direction == "Forward":
                self.include({best_feature}, task_state)
            else:
                self.exclude({best_feature}, task_state)

    def add_to_history(self, destination: str, action: str, features: Set[Variable]):
        """Add action to history"""
        self.history.append((destination, action, features))

    def step_back(self, callback: Callable = dummy_callback):
        """Restore last action from history."""
        if self.history:
            destination, action, features = self.history.pop(-1)
            set_ = getattr(self, destination)
            if action == "in":
                set_ -= features
            else:  # action == "out"
                set_ |= features
            if destination == "selected":
                # recompute scores only if selected change, not required for locked
                self.compute_feature_scores(callback)

    def run(
        self, stopping_rule: Tuple[str, Dict], task_state: Optional[TaskState] = None
    ):
        """
        Run multiple steps until stopping criteria is met or all features
        included/excluded.
        """
        if self.data is None:
            return
        learner = self.scorer.get_learner(self.data)
        stopping = Stopping(stopping_rule, self.direction, self.data, learner)


        def feature_can_be_added():
            attrs = len(self.data.domain.attributes)
            if self.direction == "Forward":
                return len(self.selected | self.locked) < attrs
            else:
                return len(self.selected - self.locked) > 0

        i = 0
        while (
            feature_can_be_added()
            and not stopping.finished(self.selected, self.scores)
            and not self.stop
        ):
            start_time = time.time()
            self.step(task_state)

            if task_state:
                task_state.set_partial_result("all")  # trigger change in the widget
                # wait only when task_state to avoid waiting when called by preprocessor
                # decrease sleep time over steps for case when many parameters are added
                sleep_time = max(0.1, 0.5 - i * 0.02)
                # if step shorter than sleep_time  wait that steps are visible in widget
                time.sleep(max(sleep_time - (time.time() - start_time), 0.01))
            i += 1

        self.stop = False

    def stop_run(self):
        """
        Currently concurrent mixin doesn't have a nice way to stop running but still
        wait until distances of current step compute. It enables only immediate stop
        which mean that feature is included but distances aren't recomputed. This method
        is called when user want to stop stepping it informs the loop in the run method
        to stop and widget's on_done method is called at the end to show new socres.
        """
        # through this variable widget tells the process to stop running
        self.stop = True


class FeatureSelectionPreprocessor(Preprocess):
    """
    Preprocessor for stepwise feature selection. It adds/removes features until
    stopping criteria is met.
    """

    def __init__(
        self,
        direction: str,
        scoring_method: str,
        validation: Tuple[str, Dict[str, float]],
        stopping_rule: Tuple[str, Dict],
        learner: Learner,
    ):
        """
        Parameters
        ----------
        direction
            The Direction of features selection. If backward the preprocessor
            will start from all features included and remove features until
            stopping criteria is met.
        scoring_method
            The scoring method to compute score differences for selecting best feature.
            For list of all supported methods check Scoring class.
        validation
            The validation method used for score computation (e.g. cross validation).
        stopping_rule
            The criteria that needs to be meet that feature adding/removing is finished.
            For list of all supported methods check Stopping class.
        learner
            The learner used to fit the model which predictions are used
            for score computing.
        """
        self.sfs = StepwiseFeatureSelection(
            direction=direction,
            scoring_method=scoring_method,
            validation=validation,
            learner=learner,
        )
        self.stopping_rule = stopping_rule

    def __call__(self, data: Table) -> Table:
        selected = set()
        if self.sfs.direction == "Backward":
            # for backward include all features and then remove until model improves
            selected = set(data.domain.attributes)
        self.sfs.set_data(data, selected)
        self.sfs.compute_feature_scores()
        self.sfs.run(self.stopping_rule)
        new_domain = Domain(
            [a for a in data.domain.attributes if a in self.sfs.selected],
            data.domain.class_vars,
            data.domain.metas,
        )
        return data.transform(new_domain)
