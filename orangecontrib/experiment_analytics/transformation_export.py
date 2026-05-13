import re
import warnings
from importlib import metadata
from typing import Optional, Sequence, Tuple

import pandas as pd
from Orange.data import (
    Domain,
    Table,
    DiscreteVariable,
    ContinuousVariable,
    StringVariable,
    TimeVariable,
    table_from_frame,
)
from Orange.preprocess import Preprocess, PreprocessorList

TRANSFORMATIONS_ATTRIBUTE = "transformations"
HTML_TABLE_STYLE = (
    "<style>th { text-align: right; vertical-align: top; } "
    "th, td {line-height: 125%}</style>"
)


def create_info_html_table(table: Sequence[Tuple]) -> str:
    table = "".join(
        f"<tr><th>{tx}: </th><td>{nu}</td></tr>" if nu else "" for tx, nu in table
    )
    return f"{HTML_TABLE_STYLE}<div><table>{table}</table></div>"


class Transformation(Preprocess):
    def __init__(self):
        # following two rows are used to know if number of rows in table
        # changed before or after the transformation
        self.rows_before = None
        self.rows_after = None
        # in all transformations except ComputeValueTransform domain is used
        # to know if any Orange/domain transformation happened after this transform
        self.domain = None

    def set_domain(self, domain: Domain):
        self.domain = domain

    def set_row_count(self, data_before: Table, data_after: Table):
        self.rows_before = len(data_before)
        self.rows_after = len(data_after)


class ComputeValueTransform(Transformation):
    """Transformation using compute_values from domain"""

    def __init__(self, domain: Domain, previous_domain: Optional[Domain] = None):
        super().__init__()
        self.domain = domain
        self.num_moved = 0
        self.num_changed = 0
        self.num_unchanged = 0
        self.num_removed = 0
        if previous_domain is not None:
            self.__compute_statistic(previous_domain)

    def __compute_statistic(self, previous_domain):
        pairs = (
            (self.domain.attributes, previous_domain.attributes),
            (self.domain.class_vars, previous_domain.class_vars),
            (self.domain.metas, previous_domain.metas),
        )
        for part_curr, part_before in pairs:
            for attr in part_curr:
                if attr in previous_domain:
                    if attr not in part_before:
                        self.num_moved += 1
                    else:
                        self.num_unchanged += 1
                else:
                    self.num_changed += 1

        len_ = len(previous_domain)
        self.num_removed = len_ - self.num_unchanged - self.num_changed - self.num_moved

    def __call__(self, data: Table) -> Table:
        return data.transform(self.domain)

    def __repr__(self):
        pairs = (
            ("Changed/added features", self.num_changed),
            ("Moved features", self.num_moved),
            ("Removed features", self.num_removed),
            ("Unchanged features", self.num_unchanged),
        )
        return f"<h4>Domain transformation</h4>{create_info_html_table(pairs)}"

    def __eq__(self, other):
        return self.domain == other.domain


class InfoTransform(Transformation):
    """
    Transformation that only stores domain for checking if any Orange
    transformation happened but do not have effect while transforming
    """

    def __init__(self, domain: Domain):
        super().__init__()
        self.domain = domain

    def __call__(self, data: Table) -> Table:
        self.__check_domains_matches(data)
        # transform to ensure order and features being in right parts of domain
        return data.transform(self.domain)

    def __check_domains_matches(self, data: Table):
        if missing := [attr.name for attr in self.domain if attr not in data.domain]:
            raise ValueError(
                f"The data are missing the following features: {', '.join(missing)}"
            )
        if additional := [attr.name for attr in data.domain if attr not in self.domain]:
            additional = ", ".join(additional)
            warnings.warn(
                "The data includes the following features that haven't been "
                f"present in the transformation workflow data: {additional}"
            )

    def __eq__(self, other):
        return self.domain == other.domain

    def __repr__(self):
        return "Domain info"


class TransformationPreprocessorList(PreprocessorList):
    DTYPES = {
        DiscreteVariable: "category",
        ContinuousVariable: float,
        StringVariable: str,
        TimeVariable: "datetime64[ns]",
    }
    ERROR_MESSAGE = (
        "Converting {attr} feature to {attr_type} failed with: '{err}'. "
        "To be able to use the provided data for the transformations, "
        "all features must be able to be transformed into the original "
        "data types used to create the transformation."
    )

    def __init__(self, preprocessors):
        super().__init__(preprocessors)
        self.__requirements = self.__get_requirements()

    @property
    def requirements(self):
        return self.__requirements

    def from_pandas(self, df):
        table = self.__df_to_table(df)
        return self.__call__(table)

    def __df_to_table(self, df: pd.DataFrame) -> Table:
        assert isinstance(self.preprocessors[0], InfoTransform)
        domain = self.preprocessors[0].domain
        # transform dataframe columns to correct type according to saved domain
        for attr in domain:
            # transforming each variable separately to report errors per column
            # pandas do not report which column caused the error
            if attr.name in df.columns:
                type_ = self.DTYPES[type(attr)]
                try:
                    df = df.astype({attr.name: type_})
                except ValueError as ex:
                    type_ = type_ if isinstance(type_, str) else type_.__name__
                    m = self.ERROR_MESSAGE.format(attr=attr, attr_type=type_, err=ex)
                    raise ValueError(m) from ex

        # transform frame to table
        return table_from_frame(df)

    @staticmethod
    def __get_requirements():
        include = {"numpy", "orange.*", "pandas", "scikit-learn", "scipy"}
        include_regex = "|".join(include)

        packages = {}
        for dist in metadata.distributions():
            name = dist.metadata["Name"]
            if re.search(include_regex, name or "", flags=re.IGNORECASE):
                packages[name] = dist.version
        return packages


def add_transformation_to_data(
    data: Table, transformation: Optional[Transformation], previous_data: Table
):
    if TRANSFORMATIONS_ATTRIBUTE in previous_data.attributes:
        transformations = previous_data.attributes[TRANSFORMATIONS_ATTRIBUTE]
        previous_step_domain = transformations[-1].domain
        if previous_step_domain != previous_data.domain:
            # if domain after previous transformation is different from domain before
            # current transformation - it indicates that in between are some Orange
            # transformation widget
            cv = ComputeValueTransform(previous_data.domain, previous_step_domain)
            cv.set_row_count(previous_data, previous_data)
            transformations += (cv,)

        if transformation:
            # transformation is None when we want to save domain only (in export widget)
            transformation.set_domain(data.domain)
            transformation.set_row_count(previous_data, data)
            transformations += (transformation,)
        data.attributes[TRANSFORMATIONS_ATTRIBUTE] = transformations
