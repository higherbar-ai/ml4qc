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

"""Module for using machine learning classification techniques on survey data."""

from ml4qc import SurveyML
from typing import Union
import numpy as np
import pandas as pd
import sklearn as skl
from matplotlib import pyplot as plt
import seaborn as sns


class SurveyMLClassifier(SurveyML):
    """Class for using machine learning classification techniques on survey data."""

    def __init__(self, x_train_df: pd.DataFrame, y_train_df: pd.DataFrame, x_predict_df: pd.DataFrame = None,
                 test_size: Union[float, int] = None, cv_when_training: bool = False,
                 random_state: Union[int, np.random.RandomState] = None, verbose: bool = None,
                 reweight_classes: bool = True):
        """
        Initialize survey data for classification using machine learning techniques.

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
        :param random_state: Fixed random state for reproducible results, otherwise None for random execution
        :type random_state: Union[int, np.random.RandomState]
        :param verbose: True to report verbose results with print() calls
        :type verbose: bool
        :param reweight_classes: True to reweight classes to account for imbalanced (skewed) data; class weights will
            be in the class_weights member variable
        :type reweight_classes: bool

        Note: Currently, only binary classification problems are supported.
        """

        # initialize base class first
        super().__init__(x_train_df, y_train_df, x_predict_df, test_size, cv_when_training, random_state, verbose)

        # confirm that we're set up for binary classification problems (the only problems currently supported)
        unique_vals = self.y_train_df.iloc[:, 0].unique()
        if self.y_train_df.shape[1] != 1 or len(unique_vals) != 2 or 0 not in unique_vals or 1 not in unique_vals:
            raise ValueError("Currently, only binary classification problems are supported, so must have a single "
                             "target column with only 0's and 1's in it.")

        # save some basic stats on the training set
        self.total_train = self.y_train_df.size
        self.pos_train = self.y_train_df.iloc[:, 0].sum()
        self.neg_train = self.total_train - self.pos_train

        # support optional class reweighting
        if reweight_classes:
            # define class weights that can help to make up for class imbalance
            # (from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data)
            self.class_weights = {0: (1/self.neg_train)*(self.total_train/2.0),
                                  1: (1/self.pos_train)*(self.total_train/2.0)}
        else:
            self.class_weights = None

        # initialize extra member variables
        self.result_y_train_predicted = None
        self.result_y_predict_predicted = None
        self.result_y_predict_predicted_proba = None
        self.result_train_accuracy = None
        self.result_train_precision = None
        self.result_train_f1 = None
        self.result_predict_accuracy = None
        self.result_predict_precision = None
        self.result_predict_f1 = None
        self.result_predict_roc_auc = None
        self.result_cv_scores = None

    def run_prediction_model(self, classifier, supports_cv: bool = True):
        """
        Execute a classification model.

        :param classifier: Classifier to use for prediction (must be sklearn estimator)
        :type classifier: Any
        :param supports_cv: False if the classifier doesn't support cross-validation (with scores including 'accuracy',
            'precision', 'f1', 'roc_auc')
        :type supports_cv: bool
        :return: Predicted classifications for the prediction set
        :rtype: Any

        In addition to the predictions that are returned, the following results can be found in member variables:

        * **result_y_train_predicted** - Predicted classifications for the training set
        * **result_y_predict_predicted** - Predicted classifications for the prediction set
        * **result_y_predict_predicted_proba** - Predicted probabilities for the training set
        * **result_train_accuracy** - Accuracy predicting within training set
        * **result_train_precision** - Precision predicting within training set
        * **result_train_f1** - F1 score for predictions within training set
        * **result_predict_accuracy** - Accuracy predicting within prediction set
        * **result_predict_precision** - Precision predicting within prediction set
        * **result_predict_f1** - F1 score for predictions within prediction set
        * **result_predict_roc_auc** - ROC AUC score for predictions within prediction set
        * **result_cv_scores** - Cross-validation scores
        """

        # first make sure we have what we need to run
        if self.x_train_preprocessed is None or self.y_train_preprocessed is None \
                or self.x_predict_preprocessed is None:
            raise ValueError("Must call preprocess_for_prediction() to preprocess data before calling "
                             "run_prediction_model().")

        # output key inputs if in verbose mode
        if self.verbose:
            print("Running prediction model...")
            print()
            print(f"  Training set: {self.x_train_preprocessed.shape} ({self.pos_train} positive)")
            print(f"Prediction set: {self.x_predict_preprocessed.shape}")

        # if requested, cross-validate training and save results
        if self.cv_when_training and supports_cv:
            if self.verbose:
                print()
                print("Cross-validating model on training set...")
                print()
            cv = skl.model_selection.RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=None)
            self.result_cv_scores = skl.model_selection.cross_validate(classifier, self.x_train_preprocessed,
                                                                       self.y_train_preprocessed, cv=cv,
                                                                       scoring=('accuracy', 'precision', 'f1',
                                                                                'roc_auc'))
        else:
            self.result_cv_scores = None

        # fit model and make predictions
        if self.verbose:
            print()
            print("Fitting model...")
            print()
        classifier.fit(self.x_train_preprocessed, self.y_train_preprocessed)
        self.result_y_train_predicted = classifier.predict(self.x_train_preprocessed)
        self.result_y_predict_predicted = classifier.predict(self.x_predict_preprocessed)
        self.result_y_predict_predicted_proba = classifier.predict_proba(self.x_predict_preprocessed)[:, 1]

        # save key stats
        self.result_train_accuracy = skl.metrics.accuracy_score(self.y_train_preprocessed,
                                                                self.result_y_train_predicted)
        self.result_train_precision = skl.metrics.precision_score(self.y_train_preprocessed,
                                                                  self.result_y_train_predicted)
        self.result_train_f1 = skl.metrics.f1_score(self.y_train_preprocessed, self.result_y_train_predicted)
        if self.y_predict_preprocessed is not None:
            self.result_predict_accuracy = skl.metrics.accuracy_score(self.y_predict_preprocessed,
                                                                      self.result_y_predict_predicted)
            self.result_predict_precision = skl.metrics.precision_score(self.y_predict_preprocessed,
                                                                        self.result_y_predict_predicted)
            self.result_predict_f1 = skl.metrics.f1_score(self.y_predict_preprocessed, self.result_y_predict_predicted)
            self.result_predict_roc_auc = skl.metrics.roc_auc_score(self.y_predict_preprocessed,
                                                                    self.result_y_predict_predicted)

        # report out automatically if in verbose mode
        if self.verbose:
            self.report_prediction_results()

        # return predictions
        return self.result_y_predict_predicted

    def report_prediction_results(self):
        """
        Report out on prediction results (after run_prediction_model()).
        """

        print("      Train accuracy: ", '{0:.2%}'.format(self.result_train_accuracy))
        print("     Train precision: ", '{0:.2%}'.format(self.result_train_precision))
        print("           Train F-1: ", '{0:.2}'.format(self.result_train_f1))

        if self.y_predict_preprocessed is not None:
            print(" Prediction accuracy: ", '{0:.2%}'.format(self.result_predict_accuracy))
            print("Prediction precision: ", '{0:.2%}'.format(self.result_predict_precision))
            print("      Prediction F-1: ", '{0:.2}'.format(self.result_predict_f1))
            print("  Test ROC_AUC Score: ", '{0:.2}'.format(self.result_predict_roc_auc))

        if self.result_cv_scores is not None:
            print()
            print("Cross validation results: ")
            print()
            for score_key, score_value in self.result_cv_scores.items():
                print(f"{score_key}: {np.mean(score_value)} (SD: {np.std(score_value)})")

        if self.y_predict_preprocessed is not None:
            skl.metrics.RocCurveDisplay.from_predictions(self.y_predict_preprocessed, self.result_y_predict_predicted)
            plt.title('ROC_AUC_Plot')
            plt.show()

            # plot predicted probabilities against the truth
            plt.figure(figsize=(15, 7))
            plt.hist(self.result_y_predict_predicted_proba[(self.y_predict_preprocessed == 0)
                                                           & (self.result_y_predict_predicted == 0)],
                     bins=50, range=(0, 1), label='True negatives', color='lightgrey')
            plt.hist(self.result_y_predict_predicted_proba[(self.y_predict_preprocessed == 0)
                                                           & (self.result_y_predict_predicted == 1)],
                     bins=50, range=(0, 1), label='False positives', color='grey')
            plt.hist(self.result_y_predict_predicted_proba[(self.y_predict_preprocessed == 1)
                                                           & (self.result_y_predict_predicted == 0)],
                     bins=50, range=(0, 1), label='False negatives', alpha=0.7, color='r')
            plt.hist(self.result_y_predict_predicted_proba[(self.y_predict_preprocessed == 1)
                                                           & (self.result_y_predict_predicted == 1)],
                     bins=50, range=(0, 1), label='True positives', alpha=0.7, color='g')
            plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
            plt.title("Predicted probabilities for prediction set")
            plt.xlabel('P(Positive)', fontsize=25)
            plt.legend(fontsize=15)
            plt.tick_params(axis='both', labelsize=25, pad=5)
            plt.show()

            # show confusion matrix
            cm = skl.metrics.confusion_matrix(self.y_predict_preprocessed, self.result_y_predict_predicted)
            names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
            counts = [value for value in cm.flatten()]
            percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
            labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names, counts, percentages)]
            labels = np.asarray(labels).reshape(2, 2)
            sns.heatmap(cm, annot=labels, cmap='Blues', fmt='')

            # summarize conditional probabilities
            tn, fp, fn, tp = cm.ravel()
            print("Test P(1) = ", '{0:.2%}'.format(self.y_predict_preprocessed.sum()
                                                   / self.y_predict_preprocessed.size))
            print("Test P(1 | predicted 1) = ", '{0:.2%}'.format(tp / (tp + fp)))
            print("Test P(1 | predicted 0) = ", '{0:.2%}'.format(fn / (tn + fn)))
            print()

            # output classification report
            print(skl.metrics.classification_report(self.y_predict_preprocessed, self.result_y_predict_predicted))

            # plot precision-recall curve
            #   calculate precision and recall
            precision, recall, thresholds = skl.metrics.precision_recall_curve(self.y_predict_preprocessed,
                                                                               self.result_y_predict_predicted_proba)
            #   create precision recall curve
            fig, ax = plt.subplots()
            ax.plot(recall, precision, color='green')
            #   add axis labels to plot
            ax.set_title('Precision-recall curve (prediction set)')
            ax.set_ylabel('Precision')
            ax.set_xlabel('Recall')
            #   add marker for 50% threshold
            halfway_index = np.argmax(thresholds >= 0.50)
            plt.axvline(recall[halfway_index], c='grey', ls=':')
            plt.axhline(precision[halfway_index], c='grey', ls=':')
            #   display plot
            plt.show()

    def report_feature_importance(self, importance_array: np.ndarray):
        """
        Report feature importance.

        :param importance_array: The appropriate importance array, depending on the classifier
        :type importance_array: np.ndarray
        """

        if self.feature_names_preprocessed is not None:
            feature_names = self.feature_names_preprocessed
        else:
            # just make up names if we don't have any
            feature_names = [*range(len(importance_array))]

        named_importance = sorted(zip(feature_names, importance_array), key=lambda x: abs(x[1]), reverse=True)
        print("Feature importance:")
        for feature_name, importance in named_importance:
            print("Feature: %s, Score: %.5f" % (feature_name, importance))
