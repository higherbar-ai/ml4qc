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
from sklearn.linear_model import LogisticRegression
from k_means_constrained import KMeansConstrained


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

    def identify_clusters(self, min_clusters: int = 2, max_clusters: int = 10, constrain_cluster_size: bool = False,
                          variance_to_retain: float = 1.0, separate_outliers: bool = True) -> pd.DataFrame:
        """
        Identify clusters in the full dataset (training+prediction together).

        :param min_clusters: Minimum number of clusters
        :type min_clusters: int
        :param max_clusters: Maximum number of clusters
        :type max_clusters: int
        :param constrain_cluster_size: True to constrain cluster size such that clusters are at least 1/2 the average
            size and at most 2x the average size
        :type constrain_cluster_size: bool
        :param variance_to_retain: Percent variance to retain, with value between 0 and 1 to use PCA for dimensionality
            reduction and 1.0 to use all features
        :type variance_to_retain: float
        :param separate_outliers: True to separate outliers into their own cluster (can help to better define other
            clusters)
        :type separate_outliers: bool
        :return: DataFrame with a cluster column that identifies clusters, indexed with the same index as the
            training and/or prediction data
        :rtype: pd.DataFrame
        """

        # confirm parameters are valid
        if min_clusters < 2 or min_clusters > max_clusters:
            raise ValueError("The min_clusters parameter must be >= 2 and <= max_clusters.")
        if max_clusters < 2 or max_clusters > 1000:
            raise ValueError("The max_clusters parameter must be >= 2 and <= 1,000.")

        # identify outliers, if we're meant to separate them
        if separate_outliers:
            if self.verbose:
                print("Identifying outliers...")
            outliers_df = self.identify_outliers()
            if self.verbose:
                print(f"{outliers_df.is_outlier.sum()} outliers detected, will be put in separate cluster (-1).")
                print()
            # drop down to only outliers, set "is_outlier" column to -1 and rename to "cluster"
            outliers_df = outliers_df[outliers_df.is_outlier == 1]
            outliers_df.rename(columns={'is_outlier': 'cluster'}, inplace=True)
            outliers_df.cluster = -1
        else:
            outliers_df = None

        # prep feature DataFrame with all data to consider
        x_all_df = pd.concat([self.x_train_df, self.x_predict_df])
        if outliers_df is not None:
            # if we're separating outliers, drop them from this dataset
            x_all_df = x_all_df[~x_all_df.index.isin(outliers_df.index)]
        x_len = len(x_all_df)

        # preprocess data: one-hot encode any categorical data, scale everything, and possibly use PCA to reduce
        transformer = ColumnTransformer(
            [('categorical', skl.preprocessing.OneHotEncoder(handle_unknown='ignore'),
              self.features_by_type["other"])], remainder='passthrough')
        if variance_to_retain < 1.0:
            preprocessing_pipeline = Pipeline(steps=[
                ('transform', transformer),
                ('scale', skl.preprocessing.StandardScaler()),
                ('reduce', PCA(n_components=variance_to_retain, svd_solver="full", random_state=self.random_state))
            ])
        else:
            preprocessing_pipeline = Pipeline(steps=[
                ('transform', transformer),
                ('scale', skl.preprocessing.StandardScaler())
            ])
        if self.verbose:
            print(f"  Starting features shape: {x_all_df.shape}")
        x_data = preprocessing_pipeline.fit_transform(x_all_df)
        if self.verbose:
            print(f"     Final features shape: {x_data.shape}")
            print()

        # generate clusters for range of n_clusters options, keeping best set of labels based on silhouette coefficient
        if self.verbose:
            print(f"Choosing best silhouette coefficient for n_clusters between {min_clusters} and {max_clusters}...")
        best_score = -1
        best_labels = None
        for n_clusters in range(min_clusters, max_clusters+1):
            if constrain_cluster_size:
                # insist on reasonably-balanced cluster sizes: no clusters smaller than 1/2 average cluster size and no
                # clusters larger than 2x average cluster size
                min_cluster = int((x_len / n_clusters) / 2)
                max_cluster = min(int((x_len / n_clusters) * 2), x_len)
            else:
                min_cluster = None
                max_cluster = None
            labels = KMeansConstrained(n_clusters=n_clusters, size_min=min_cluster, size_max=max_cluster,
                                       random_state=self.random_state).fit_predict(x_data)
            score = skl.metrics.silhouette_score(x_data, labels)
            if best_score == -1 or score > best_score:
                best_labels = labels
                best_score = score
            if self.verbose:
                print()
                print(f"Silhouette coefficient for {n_clusters} clusters: {score}")
                print(pd.DataFrame(labels).iloc[:, 0].value_counts())

        # take best set of cluster labels, and merge in outliers if they'd been separated
        result_df = pd.DataFrame(best_labels, columns=['cluster']).set_index(x_all_df.index.values)
        if outliers_df is not None:
            result_df = pd.concat([result_df, outliers_df])

        return result_df

    def benchmark_by_category(self, category_df: pd.DataFrame, benchmark_categories: list[str], method: str = 'knn',
                              n_nearest_neighbors: int = 10, reg_strength: float = 0.0001,
                              variance_to_retain: float = 1.0) -> pd.DataFrame:
        """
        Benchmark by category (e.g., by enumerator) using the full dataset (training+prediction together). Uses
        classification method to score observations as more or less like the identified category or categories
        to benchmark against (e.g., one or more star enumerators).

        :param category_df: Category column for benchmarking, in a Pandas DataFrame indexed with the same index as the
            training and prediction data used to initialize the object
        :type category_df: pd.DataFrame
        :param benchmark_categories: List of specific categories to benchmark against (e.g., one or more star enumerator
            IDs, if benchmarking by enumerator)
        :type benchmark_categories: list[str]
        :param method: Method to use for scoring ('knn' for K nearest neighbors, 'logistic' for logistic regression)
        :type method: str
        :param n_nearest_neighbors: If method is 'knn', number of nearest neighbors to consider (not including self);
            the largest this is, the more it skews toward categories with more observations in the dataset
        :type n_nearest_neighbors: int
        :param reg_strength: If method is 'logistic', C value for regularization strength to use for L2 regularization;
            given that we're classifying within the training set, a larger value will tend toward a perfect fit
        :type reg_strength: float
        :param variance_to_retain: Percent variance to retain, with value between 0 and 1 to use PCA for dimensionality
            reduction and 1.0 to use all features
        :type variance_to_retain: float
        :return: DataFrame with category-specific scores, sorted highest first
        :rtype: pd.DataFrame
        """

        # prep target DataFrame with observations in benchmark categories coded as 1 vs. 0
        category_df = category_df.sort_index()
        category_df["is_benchmark"] = category_df.apply(lambda row: (1 if row.iloc[0] in benchmark_categories else 0),
                                                        axis=1)
        target_df = pd.DataFrame(category_df["is_benchmark"]).sort_index()
        if self.verbose:
            print(target_df.is_benchmark.value_counts())
            print()
            print(f"Target base rate: {(len(target_df[target_df.is_benchmark == 1]) / len(target_df)) * 100:.2f}%")
            print()

        # calculate predicted probabilities for all observations
        if method == 'knn':
            classifier = skl.neighbors.KNeighborsClassifier(n_neighbors=n_nearest_neighbors + 1,
                                                            weights=SurveyML.inverse_distance_ignoring_closest)
        elif method == 'logistic':
            classifier = LogisticRegression(random_state=self.random_state, max_iter=2000, penalty="l2", C=reg_strength)
        else:
            raise ValueError(f"Unsupported method: {method}")
        predictions, predicted_proba = self._classify_with_all_data(target_df=target_df, classifier=classifier,
                                                                    variance_to_retain=variance_to_retain)

        # summarize predicted probabilities by enumerator
        result_df = pd.DataFrame(category_df.iloc[:, 0])
        result_df["score"] = predicted_proba[:, 1]

        return result_df.groupby(result_df.columns[0])[['score']].mean().sort_values("score", ascending=False)

    def classify_by_category(self, category_df: pd.DataFrame, method: str = 'knn', n_nearest_neighbors: int = 10,
                             reg_strength: float = 0.0001, variance_to_retain: float = 1.0) -> pd.DataFrame:
        """
        Classify by category (e.g., by enumerator) using the full dataset (training+prediction together).

        :param category_df: Category column for classification, in a Pandas DataFrame indexed with the same index as the
            training and prediction data used to initialize the object
        :type category_df: pd.DataFrame
        :param method: Method to use for classification ('knn' for K nearest neighbors, 'logistic' for logistic
            regression)
        :type method: str
        :param n_nearest_neighbors: If method is 'knn', number of nearest neighbors to consider (not including self);
            the largest this is, the more it skews toward categories with more observations in the dataset
        :param reg_strength: If method is 'logistic', C value for regularization strength to use for L2 regularization;
            given that we're classifying within the training set, a larger value will tend toward a perfect fit
        :type reg_strength: float
        :param variance_to_retain: Percent variance to retain, with value between 0 and 1 to use PCA for dimensionality
            reduction and 1.0 to use all features
        :type variance_to_retain: float
        :return: DataFrame with category predictions for each observation in the dataset
        :rtype: pd.DataFrame
        """

        # prep target DataFrame
        target_df = category_df.sort_index()

        # calculate predicted categories for all observations
        if method == 'knn':
            classifier = skl.neighbors.KNeighborsClassifier(n_neighbors=n_nearest_neighbors + 1,
                                                            weights=SurveyML.inverse_distance_ignoring_closest)
        elif method == 'logistic':
            classifier = LogisticRegression(random_state=self.random_state, max_iter=2000, penalty="l2", C=reg_strength)
        else:
            raise ValueError(f"Unsupported method: {method}")
        predictions, predicted_proba = self._classify_with_all_data(target_df=target_df, classifier=classifier,
                                                                    variance_to_retain=variance_to_retain)

        # combine predictions with original categories and return
        result_df = pd.DataFrame(category_df)
        result_df["predicted_category"] = predictions
        return result_df

    def _classify_with_all_data(self, target_df: pd.DataFrame, classifier,
                                variance_to_retain: float = 1.0) -> (np.ndarray, np.ndarray):
        """
        Run classifier using the full dataset (training+prediction together).

        :param target_df: Target classes for classification task
        :type target_df: pd.DataFrame
        :param classifier: Classifier to use
        :type classifier: Any
        :param variance_to_retain: Percent variance to retain, with value between 0 and 1 to use PCA for dimensionality
            reduction and 1.0 to use all features
        :type variance_to_retain: float
        :return: Predicted label array and class probability array (probabilities ordered by classes in lexicographic
            order)
        :rtype: (np.ndarray, np.ndarray)
        """

        # prep feature DataFrame with all data, confirm our indexes match
        x_all_df = pd.concat([self.x_train_df, self.x_predict_df]).sort_index()
        if not np.array_equal(np.array(x_all_df.index), np.array(target_df.index)):
            raise ValueError("The category_df parameter must include all items in training and prediction DataFrames, "
                             "with the same index values.")

        # preprocess data: one-hot encode any categorical data, scale everything, and possibly use PCA to reduce
        transformer = ColumnTransformer(
            [('categorical', skl.preprocessing.OneHotEncoder(handle_unknown='ignore'),
              self.features_by_type["other"])], remainder='passthrough')
        if variance_to_retain < 1.0:
            preprocessing_pipeline = Pipeline(steps=[
                ('transform', transformer),
                ('scale', skl.preprocessing.StandardScaler()),
                ('reduce', PCA(n_components=variance_to_retain, svd_solver="full", random_state=self.random_state))
            ])
        else:
            preprocessing_pipeline = Pipeline(steps=[
                ('transform', transformer),
                ('scale', skl.preprocessing.StandardScaler())
            ])
        if self.verbose:
            print(f"  Starting features shape: {x_all_df.shape}")
        x_data = preprocessing_pipeline.fit_transform(x_all_df)
        if self.verbose:
            print(f"     Final features shape: {x_data.shape}")
            print()

        # fit simple K-nearest-neighbors model with K neighbors averaged by distance (where K includes oneself, so
        # effectively K-1 neighbors once self is weighted down to 0)
        if self.verbose:
            print("Fitting model...")
            print()
        classifier.fit(x_data, target_df.values.ravel())

        # return predicted probabilities
        return classifier.predict(x_data), classifier.predict_proba(x_data)

    @staticmethod
    def inverse_distance_ignoring_closest(distances):
        """
        Distance weighting function that sets closest distance to 0.0 and otherwise inverses distances. Useful for
        cases where you're using nearest-neighbor methods but predicting from the training set (in which the closest
        match will be yourself).

        :param distances: Array of distances (assumed 1D or 2D)
        :type distances: Any
        :return: Array of weights with 0.0 for smallest distance in each set, otherwise inverses of weights (same shape
            as the distances array passed in)
        :rtype: Any
        """

        # always convert distances to floats
        float_distances = distances.astype(float)
        # drop smallest distance to 0 in order to ignore closest neighbor
        if float_distances.ndim > 1:
            float_distances[np.arange(len(float_distances)), float_distances.argmin(axis=1)] = 0.0
        else:
            float_distances[float_distances.argmin()] = 0.0
        # weight non-zero distances as reciprocal
        weights = np.reciprocal(float_distances, out=float_distances, where=float_distances != 0.0)
        return weights

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
