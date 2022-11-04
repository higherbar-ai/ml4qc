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

from surveydata.surveyctoplatform import SurveyCTOPlatform
from surveydata.storagesystem import StorageSystem
import pandas as pd
import requests
import datetime
from urllib.parse import unquote_plus


class SurveyCTOMLPlatform(SurveyCTOPlatform):
    """SurveyCTO survey data platform implementation, extended for ML-based quality control."""

    # define constants
    REVIEW_STATUS_VALUE = {"none": "NONE", "approved": "REJECTED", "rejected": "REJECTED"}
    REVIEW_STATUS_LABEL = {"none": "set to pending", "approved": "approved", "rejected": "rejected"}
    QUALITY_VALUE = {"good": "ct_good", "okay": "ct_okay", "poor": "ct_poor", "fake": "ct_fake"}
    QUALITY_LABEL = {"good": "GOOD", "okay": "OKAY", "poor": "POOR", "fake": "FAKE"}

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

        If you're not going to call sync_data() or review_submissions(), you don't need to supply any of the parameters
            to this constructor.
        """

        # call parent class version
        super().__init__(server, username, password, formid, private_key)

    def sync_data(self, storage: StorageSystem, attachment_storage: StorageSystem = None,
                  no_attachments: bool = False, review_statuses: list = None) -> list:
        """
        Sync survey data to storage system.

        :param storage: Storage system for submissions (and attachments, if supported and other options don't override)
        :type storage: StorageSystem
        :param attachment_storage: Separate storage system for attachments (only if needed)
        :type attachment_storage: StorageSystem
        :param no_attachments: True to not sync attachments
        :type no_attachments: bool
        :param review_statuses: List of review statuses to include (any combo of "approved", "pending", "rejected";
            if not specified, syncs only approved submissions)
        :type review_statuses: list
        :return: List of new submissions stored (submission ID strings)
        :rtype: list
        """

        # call parent class version
        return super().sync_data(storage, attachment_storage, no_attachments, review_statuses)

    def update_submissions(self, submission_updates: list):
        """
        Submit one or more submission updates, including reviews, classifications, and/or comments.

        :param submission_updates: List of dictionaries with one per update; each should include values for
            "submissionID"; "reviewStatus" ("none", "approved", or "rejected"); "qualityClassification" ("good",
            "okay", "poor", or "fake"); and/or "comment" (custom text)
        :type submission_updates: list

        Warning: this method uses an undocumented SurveyCTO API that may break in future SurveyCTO releases.
        """

        # assemble review bundle from passed updates
        review_bundle = []
        timestamp = int(datetime.datetime.timestamp(datetime.datetime.now()) * 1000)
        for update in submission_updates:
            # first confirm that update looks valid
            has_valid_subid = ("submissionID" in update and update["submissionID"])
            has_valid_review_status = ("reviewStatus" in update and update["reviewStatus"]
                                       and update["reviewStatus"] in self.REVIEW_STATUS_VALUE)
            has_valid_quality = ("qualityClassification" in update and update["qualityClassification"]
                                 and update["qualityClassification"] in self.QUALITY_VALUE)
            has_valid_comment = ("comment" in update and update["comment"])
            if not has_valid_subid:
                raise ValueError("Must supply submissionID value within each update dict.")
            if not has_valid_review_status and not has_valid_quality and not has_valid_comment:
                raise ValueError("Each update dict must include at least a valid reviewStatus, qualityClassification, "
                                 "or comment.")
            if "reviewStatus" in update and not has_valid_review_status:
                raise ValueError("Invalid reviewStatus included in update dict: " + update["reviewStatus"])
            if "qualityClassification" in update and not has_valid_quality:
                raise ValueError("Invalid qualityClassification included in update dict: "
                                 + update["qualityClassification"])

            # build xReview dictionary
            xreview = {"instanceId": update["submissionID"]}
            comments = []
            if "comment" in update and update["comment"]:
                comments.append({"text": update['comment'], "type": "USER", "creationDate": timestamp})
            if has_valid_review_status and has_valid_quality:
                xreview["classTagUpdate"] = self.QUALITY_VALUE[update["qualityClassification"]]
                xreview["statusUpdate"] = self.REVIEW_STATUS_VALUE[update["reviewStatus"]]
                comments.append({"text": f"[ Submission {self.REVIEW_STATUS_LABEL[update['reviewStatus']]} via API. "
                                         f"Classified as {self.QUALITY_LABEL[update['qualityClassification']]}. ]",
                                 "type": "SYSTEM", "creationDate": timestamp})
            elif has_valid_review_status:
                xreview["statusUpdate"] = self.REVIEW_STATUS_VALUE[update["reviewStatus"]]
                comments.append({"text": f"[ Submission {self.REVIEW_STATUS_LABEL[update['reviewStatus']]} via API. ]",
                                 "type": "SYSTEM", "creationDate": timestamp})
            elif has_valid_quality:
                xreview["classTagUpdate"] = self.QUALITY_VALUE[update["qualityClassification"]]
                comments.append({"text": f"[ Classified as {self.QUALITY_LABEL[update['qualityClassification']]} "
                                         f"via API. ]",
                                 "type": "SYSTEM", "creationDate": timestamp})
            xreview["comments"] = comments

            # add to review bundle
            review_bundle.append({"xReview": xreview, "lastReviewDate": timestamp})

        # authenticate with the server, raising any errors as exceptions
        session, headers = self._authenticate_via_login()

        # post review bundle to the server
        response = session.post(f"https://{self.server}.surveycto.com/forms/{self.formid}/save-reviews",
                                cookies=session.cookies, headers=headers, json=review_bundle)
        response.raise_for_status()

    def _authenticate_via_login(self) -> (requests.Session, dict):
        """
        Authenticate with SurveyCTO server via interactive login process.

        :return: Tuple with HTTP session (with session cookies) and dict of headers to use for subsequent requests
        :rtype: (requests.Session, dict)
        """

        # fire an exception if we haven't been initialized for connections with the server
        if not self.server or self.creds is None:
            raise ValueError("SurveyCTOMLPlatform not initialized with parameters sufficient for connecting to server "
                             "(server, username, password).")

        # begin login sequence with fresh session, raising errors as exceptions
        session = requests.session()
        response = session.head(f"https://{self.server}.surveycto.com/index.html")
        response.raise_for_status()
        headers = {"X-csrf-token": response.headers["X-csrf-token"]}

        # attempt the actual login and update CSRF token
        response = session.post(f"https://{self.server}.surveycto.com/login", cookies=session.cookies,
                                headers=headers,
                                data={"username": self.creds.username, "password": self.creds.password})
        response.raise_for_status()
        if "login_failure" in response.headers and response.headers["login_failure"]:
            raise ValueError("Invalid server name or login credentials. Error from SurveyCTO: " + unquote_plus(
                response.headers["login_failure"]))
        headers = {"X-csrf-token": response.headers["X-csrf-token"]}

        return session, headers

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
        return super(SurveyCTOMLPlatform, SurveyCTOMLPlatform).get_submissions_df(storage)

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
        return super(SurveyCTOMLPlatform, SurveyCTOMLPlatform).get_text_audit_df(storage, location_string,
                                                                                 location_strings)

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
        return super(SurveyCTOMLPlatform, SurveyCTOMLPlatform)._load_text_audit(storage, location_string)
