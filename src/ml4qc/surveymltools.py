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

"""Tools for using machine learning techniques on survey data."""

import numpy as np
import pandas as pd


class SurveyMLTools(object):
    """Tools for using machine learning techniques on survey data."""

    @staticmethod
    def columns_by_type(df: pd.DataFrame) -> dict:
        """
        Get column lists by data type.

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
