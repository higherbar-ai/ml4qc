#  Copyright (c) 2022 Orange Chair Labs LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Base module for using machine learning techniques on survey data."""

from typing import Union
import numpy as np
from numpy.random import RandomState
import pandas as pd
import sklearn as skl
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class SurveyML(object):
    """Base class for using machine learning techniques on survey data."""

    def __init__(self, x_train_df: pd.DataFrame, y_train_df: pd.DataFrame, x_predict_df: pd.DataFrame = None,
                 test_size: Union[float, int] = None, cv_when_training: bool = False, n_jobs: int = -2,
                 random_state: Union[int, RandomState] = None, verbose: bool = None):
        """
        Initialize survey data for machine learning.

        :param x_train_df: Features for training dataset
        :type x_train_df: pd.DataFrame
        :param y_train_df: Target(s) for training dataset
        :type y_train_df: pd.DataFrame
        :param x_predict_df: Prediction dataset (required unless test_size used to take test set from training set)
        :type x_predict_df: pd.DataFrame
        :param test_size: Float (0, 1) for proportion of training dataset to use for testing; int for number of
            training rows to use for testing; otherwise None to specify prediction set manually in x_predict_df
        :type test_size: Union[float, int]
        :param cv_when_training: True to cross-validate when training models
        :type cv_when_training: bool
        :param n_jobs: Number of parallel jobs to run during cross-validation (-1 for as many jobs as CPU's, -2 to
            leave one CPU free)
        :type n_jobs: int
        :param random_state: Fixed random state for reproducible results, otherwise None for random execution
        :type random_state: Union[int, RandomState]
        :param verbose: True to report verbose results with print() calls
        :type verbose: bool
        """

        # ensure that we have everything we need
        if x_predict_df is None and test_size is None:
            raise ValueError("Must specify either x_predict_df or test_size in order to define a prediction set.")

        # remember all parameters in member vars
        if x_predict_df is None:
            # split training set into training and test sets
            self.x_train_df, self.x_predict_df, \
                self.y_train_df, self.y_predict_df = train_test_split(x_train_df, y_train_df,
                                                                      test_size=test_size,
                                                                      random_state=random_state)
        else:
            self.x_train_df = x_train_df
            self.y_train_df = y_train_df
            self.x_predict_df = x_predict_df
            self.y_predict_df = None

        self.test_size = test_size
        self.cv_when_training = cv_when_training
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # initialize other members
        self.preprocessing_pipeline = None
        self.x_train_preprocessed = None
        self.y_train_preprocessed = None
        self.x_predict_preprocessed = None
        self.y_predict_preprocessed = None
        self.num_features = 0
        self.feature_names_preprocessed = None

        # organize features by data type
        self.features_by_type = self.columns_by_type(self.x_train_df)
        if self.verbose:
            for dtype in self.features_by_type:
                print(f"{dtype} features: {len(self.features_by_type[dtype])}")

        # raise exception on disallowed data types
        if len(self.features_by_type["datetime"]) > 0:
            raise ValueError("datetime features not supported in SurveyML.")

    def preprocess_for_prediction(self, pca: float = None, custom_pipeline: Pipeline = None):
        """
        Preprocess data for prediction.

        :param pca: If not None, float between 0 and 1 for amount of variance to retain via PCA dimensionality reduction
        :type pca: float
        :param custom_pipeline: Custom preprocessing pipeline, if any (overrides pca parameter)
        :type custom_pipeline: Pipeline
        """

        # set up preprocessing pipeline
        if custom_pipeline is not None:
            self.preprocessing_pipeline = custom_pipeline
            transformer = None
        else:
            if pca is not None:
                # for dimensionality reduction: one-hot encode any categorical data, then scale everything
                transformer = ColumnTransformer(
                    [('categorical', skl.preprocessing.OneHotEncoder(handle_unknown='ignore'),
                      self.features_by_type["other"])], remainder='passthrough')
                self.preprocessing_pipeline = Pipeline(steps=[
                    ('transform', transformer),
                    ('scale', skl.preprocessing.StandardScaler()),
                    ('reduce', PCA(n_components=pca, svd_solver="full", random_state=self.random_state))
                ])
            else:
                # for direct use: leave binary and unit-interval data as-is, rescale other numeric data,
                #                 one-hot encode categorical data
                transformer = ColumnTransformer(
                    [('numeric_other', skl.preprocessing.MinMaxScaler(), self.features_by_type["numeric_other"]),
                     ('categorical', skl.preprocessing.OneHotEncoder(handle_unknown='ignore'),
                      self.features_by_type["other"])], remainder='passthrough')

                self.preprocessing_pipeline = Pipeline(steps=[
                    ('transform', transformer)
                ])

        if self.verbose:
            print(f"  Starting training set shape: {self.x_train_df.shape}")
            print(f"Starting prediction set shape: {self.x_predict_df.shape}")

        # perform preprocessing
        self.x_train_preprocessed = self.preprocessing_pipeline.fit_transform(self.x_train_df)
        self.y_train_preprocessed = self.y_train_df.values.ravel()
        self.x_predict_preprocessed = self.preprocessing_pipeline.transform(self.x_predict_df)
        if self.y_predict_df is None:
            self.y_predict_preprocessed = None
        else:
            self.y_predict_preprocessed = self.y_predict_df.values.ravel()

        # record final feature count, and names if we can
        self.num_features = self.x_train_preprocessed.shape[1]
        if pca is None and transformer is not None:
            self.feature_names_preprocessed = transformer.get_feature_names_out()
        else:
            # no feature names available if using PCA to reduce dimensions
            self.feature_names_preprocessed = None

        if self.verbose:
            print(f"     Final training set shape: {self.x_train_preprocessed.shape}")
            print(f"   Final prediction set shape: {self.x_predict_preprocessed.shape}")

    def identify_outliers(self, contamination: float = None) -> pd.DataFrame:
        """
        Identify outliers in the full dataset (training+prediction together).

        :param contamination: Proportion (0,1) of dataset that should be considered an outlier, or None for auto
        :type contamination: float
        :return: DataFrame with an is_outlier column that is True for outliers and False otherwise
        :rtype: pd.DataFrame
        """

        # pool all data, one-hot encode categorical data, otherwise leave everything alone/unscaled
        x_all_df = pd.concat([self.x_train_df, self.x_predict_df])
        transformer = skl.compose.ColumnTransformer(
            [('categorical', skl.preprocessing.OneHotEncoder(handle_unknown='ignore'),
              self.features_by_type["other"])], remainder='passthrough')
        x_all = transformer.fit_transform(x_all_df)

        # identify outliers
        if_classifier = IsolationForest(contamination="auto" if contamination is None else contamination,
                                        random_state=self.random_state)
        outlier_df = pd.DataFrame(if_classifier.fit_predict(x_all) == -1,
                                  columns=['is_outlier']).set_index(x_all_df.index.values)

        # use 1 and 0 vs. True and False
        outlier_df["is_outlier"] = outlier_df["is_outlier"].astype(int)

        if self.verbose:
            print(f"Outliers: {outlier_df.is_outlier.sum()} ({outlier_df.is_outlier.mean() * 100}%)")

        return outlier_df

    def features_by_type(self) -> dict:
        """
        Get features by data type.

        :return: Dictionary with six lists of column names: "numeric", "numeric_binary", "numeric_unit_interval",
            "numeric_other", "datetime", "other"
        :rtype: dict
        """

        return self.columns_by_type(self.x_train_df)

    @staticmethod
    def columns_by_type(df: pd.DataFrame) -> dict:
        """
        Get DataFrame columns by data type.

        :param df: DataFrame with columns to categorize
        :type df: pd.DataFrame
        :return: Dictionary with six lists of column names: "numeric", "numeric_binary", "numeric_unit_interval",
            "numeric_other", "datetime", "other"
        :rtype: dict
        """

        # initialize our dict for return, with main column categories
        cols_by_type = {"numeric": df.select_dtypes(include=['number']).columns.values,
                        "numeric_binary": [],
                        "numeric_unit_interval": [],
                        "numeric_other": [],
                        "datetime": df.select_dtypes(include=['datetime']).columns.values,
                        "other": df.select_dtypes(exclude=['number', 'datetime']).columns.values}

        # then add subcategories for numeric columns
        for col in cols_by_type["numeric"]:
            if np.isin(df[col].dropna().unique(), [0, 1]).all():
                cols_by_type["numeric_binary"].append(col)
            elif df[col].min() >= 0 and df[col].max() <= 1:
                cols_by_type["numeric_unit_interval"].append(col)
            else:
                cols_by_type["numeric_other"].append(col)

        return cols_by_type
