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

"""Extended version of the surveydata.SurveyCTOPlatform class for ML-based quality control."""

from surveydata.surveyplatform import SurveyPlatform
from surveydata.surveyctoplatform import SurveyCTOPlatform
from surveydata.storagesystem import StorageSystem
import pandas as pd


class SurveyCTOMLPlatform(SurveyCTOPlatform):
    """SurveyCTO survey data platform implementation, extended for ML-based quality control."""

    def __init__(self, server: str = "", username: str = "", password: str = "", formid: str = "",
                 private_key: str = ""):
        """
        Initialize SurveyCTO for access to survey data.

        :param server: SurveyCTO server name (like "use", without the https prefix or .surveycto.com suffix)
        :type server: str
        :param username: Email address for API access
        :type username: str
        :param password: Password for API access
        :type password: str
        :param formid: SurveyCTO form ID
        :type formid: str
        :param private_key: Full text of private key, if using encryption
        :type private_key: str

        If you're not going to call sync_data(), you don't need to supply any of the parameters to this constructor.
        """

        # call parent class version
        super().__init__()

    def sync_data(self, storage: StorageSystem, attachment_storage: StorageSystem = None,
                  no_attachments: bool = False) -> list:
        """
        Sync survey data to storage system.

        :param storage: Storage system for submissions (and attachments, if supported and other options don't override)
        :type storage: StorageSystem
        :param attachment_storage: Separate storage system for attachments (only if needed)
        :type attachment_storage: StorageSystem
        :param no_attachments: True to not sync attachments
        :type no_attachments: bool
        :return: List of new submissions stored (submission ID strings)
        :rtype: list
        """

        # call parent class version
        return super().sync_data(storage, attachment_storage, no_attachments)

    @staticmethod
    def get_submissions_df(storage: StorageSystem) -> pd.DataFrame:
        """
        Get all submission data from storage, organized into a Pandas DataFrame and optimized based on the platform.

        :param storage: Storage system for submissions
        :type storage: StorageSystem
        :return: Pandas DataFrame containing all submissions currently in storage
        :rtype: pandas.DataFrame
        """

        # call parent class version
        return super().get_submissions_df(storage)

    @staticmethod
    def get_text_audit_df(storage: StorageSystem, location_string: str = "",
                          location_strings: pd.Series = None) -> pd.DataFrame:
        """
        Get one or more text audits from storage, organized into a Pandas DataFrame.

        :param storage: Storage system for attachments
        :type storage: StorageSystem
        :param location_string: Location string of single text audit to load
        :type location_string: str
        :param location_strings: Series of location strings of text audits to load
        :type location_strings: pandas.Series
        :return: DataFrame with either the single text audit contents or all text audit contents indexed by Series index
        :rtype: pandas.DataFrame

        Pass either a single location_string or a Series of location_strings.
        """


        # call parent class version
        return super().get_text_audit_df(storage, location_string, location_strings)

    @staticmethod
    def _load_text_audit(storage: StorageSystem, location_string: str) -> pd.DataFrame:
        """
        Load a single text audit file from storage to a Pandas DataFrame.

        :param storage: Storage system for attachments
        :type storage: StorageSystem
        :param location_string: Location string of single text audit to load
        :type location_string: str
        :return: DataFrame with the single text audit contents
        :rtype: pandas.DataFrame
        """

        # call parent class version
        return super()._load_text_audit(storage, location_string)
