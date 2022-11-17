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
import pandas as pd
import numpy as np
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
    def process_text_audits(ta_df: pd.DataFrame, start_times: pd.Series = None, end_times: pd.Series = None,
                            data_tz: datetime.timezone = None, collection_tz: datetime.timezone = None) -> pd.DataFrame:
        """
        Process text audits by summarizing, transforming, and reshaping into a single row per submission.

        :param ta_df: DataFrame with raw text audit data, typically from get_text_audit_df()
        :type ta_df: pd.DataFrame
        :param start_times: Pandas Series with a starting date and time for each submission (indexed by submission ID)
        :type start_times: pd.Series
        :param end_times: Pandas Series with an ending date and time for each submission (indexed by submission ID)
        :type end_times: pd.Series
        :param data_tz: Timezone of timestamps in start_times and end_times
        :type data_tz: datetime.timezone
        :param collection_tz: Timezone of data collection
        :type collection_tz: datetime.timezone
        :return: Pandas DataFrame, indexed by submission ID, with summary details as well as field-by-field visit
            summaries
        :rtype: pd.DataFrame

        The returned DataFrame is indexed by submission ID and includes the following columns:

        * **ta_duration_total** - Total duration spent in form fields (ms); feature engineering recommendation: divide
          by max to rescale to 0-1
        * **ta_duration_mean** - Mean duration spent in form fields (ms); feature engineering recommendation: divide by
          max to rescale to 0-1
        * **ta_duration_sd** - Standard deviation of duration spent in form fields (ms); feature engineering
          recommendation: divide by max to rescale to 0-1
        * **ta_duration_max** - Max duration spent in form fields (ms); feature engineering recommendation: divide by
          max to rescale to 0-1
        * **ta_fields** - Number of fields visited; feature engineering recommendation: divide by max to rescale to 0-1
        * **ta_time_in_fields** - Percent of overall calendar time spent in fields; feature engineering recommendation:
          leave as 0-1 scale
        * **ta_sessions** - Number of form-filling sessions (always 1 unless eventlog-level text audit data); feature
          engineering recommendation: divide by max to rescale to 0-1
        * **ta_pct_revisits** - Percent of field visits that are revisits (always 0 unless eventlog-level text audit
          data); feature engineering recommendation: leave as 0-1 scale
        * **ta_start_dayofweek** - Day of week submission started (0 for Sunday, only available if eventlog text audit
          data or timezone information supplied); feature engineering recommendation: one-hot encode
        * **ta_start_hourofday** - Hour of day submission started (only available if eventlog text audit data or
          timezone information supplied); feature engineering recommendation: one-hot encode
        * **ta_field_x_visited** - 1 if field x visited, otherwise 0; feature engineering recommendation: leave as 0-1
          scale, fill missing with 0
        * **ta_field_x_visit_y_start** - When field x was visited the yth time divided by the highest field start time,
          otherwise 0; feature engineering recommendation: leave as 0-1 scale, fill missing with 0
        * **ta_field_x_visit_y_duration** - Time spent on field x the yth time it was visited divided by the highest
          field duration, otherwise 0; feature engineering recommendation: leave as 0-1 scale, fill missing with 0
        """

        # start list of dictionaries, one for each submission
        summaries = []
        # get list of submission IDs and process each in turn
        submission_ids = ta_df.index.unique()
        for subid in submission_ids:
            # subset out the submission's text audit data
            sub_ta_df = ta_df.loc[subid, :]
            # auto-detect text audit type for the submission (eventlog vs. traditional)
            if "duration_ms" in sub_ta_df.columns.values and sub_ta_df["duration_ms"].sum() > 0:
                eventlog = True
                duration_field = "duration_ms"
                duration_ms = sub_ta_df[duration_field].sum()
                # take form start and end times from text audit data
                start_time = sub_ta_df["device_time"].iloc[0]
                end_time = sub_ta_df["device_time"].iloc[-1]
                # since device_time includes timezone, we're confident of the collection timezone
                #   (note, though, that all web submissions are recorded in UTC, which may be misleading)
                tz_confident = True
            else:
                eventlog = False
                duration_field = "duration_s"
                duration_ms = sub_ta_df[duration_field].sum() * 1000
                # take form start and end times from passed-in form data, if available
                if start_times is not None and subid in start_times and end_times is not None and subid in end_times:
                    # record times with collection timezone if possible
                    if data_tz is not None and collection_tz is not None:
                        start_time = start_times[subid].tz_localize(data_tz).tz_convert(collection_tz)
                        end_time = end_times[subid].tz_localize(data_tz).tz_convert(collection_tz)
                        tz_confident = True
                    else:
                        start_time = start_times[subid]
                        end_time = end_times[subid]
                        tz_confident = False
                else:
                    start_time = None
                    end_time = None
                    tz_confident = False

            # populate a new dictionary for the current submission
            summary = {SurveyCTOPlatform.ID_FIELD: subid,
                       "ta_duration_total": duration_ms,
                       "ta_duration_mean": sub_ta_df[duration_field].mean() * 1000 if not eventlog
                       else sub_ta_df[duration_field].mean(),
                       "ta_duration_sd": sub_ta_df[duration_field].std() * 1000 if not eventlog
                       else sub_ta_df[duration_field].std(),
                       "ta_duration_max": sub_ta_df[duration_field].max() * 1000 if not eventlog
                       else sub_ta_df[duration_field].max(),
                       "ta_time_in_fields": np.NaN if start_time is None
                       else (duration_ms / 1000) / (end_time - start_time).seconds,
                       "ta_fields": sub_ta_df["field"].nunique(),
                       "ta_sessions": 1 if not eventlog else 1 + len(sub_ta_df[sub_ta_df["event"] == "Reopen form"]),
                       "ta_pct_revisits": 0 if not eventlog else 1 - (sub_ta_df["field"].nunique() /
                                                                      len(sub_ta_df[sub_ta_df["event"]
                                                                                    == "Visit field"])),
                       "ta_start_dayofweek": "" if not tz_confident else str(start_time.weekday()),
                       "ta_start_hourofday": "" if not tz_confident else str(start_time.hour)}

            # if there are no durations for a submission, fill missing values with 0.0
            if pd.isna(summary["ta_duration_mean"]):
                summary["ta_duration_mean"] = 0.0
                summary["ta_duration_sd"] = 0.0
                summary["ta_duration_max"] = 0.0

            for field in sub_ta_df["field"].dropna().unique():
                # strip+escape fieldname to create a version for DataFrame column names
                df_fieldname = field.strip().replace(' ', '_').replace('[', '_').replace(']', '').replace('/', '_')

                # subset down to text audit records for this field
                field_df = sub_ta_df[sub_ta_df["field"] == field]

                # populate field-specific summary columns
                summary[f"ta_field_{df_fieldname}_visited"] = 1
                for index in range(len(field_df)):
                    row = field_df.iloc[index]
                    if eventlog:
                        summary[f"ta_field_{df_fieldname}_visit_{index + 1}_start"] = \
                            row["form_time_ms"] / sub_ta_df["form_time_ms"].max()
                        summary[f"ta_field_{df_fieldname}_visit_{index + 1}_duration"] = \
                            row[duration_field] / summary["ta_duration_max"]
                    else:
                        summary[f"ta_field_{df_fieldname}_visit_{index + 1}_start"] = \
                            row["visited_s"] / sub_ta_df["visited_s"].max()
                        summary[f"ta_field_{df_fieldname}_visit_{index + 1}_duration"] = \
                            (row[duration_field] * 1000) / summary["ta_duration_max"]

            # add the current submission to the list of summaries
            summaries += [summary]

        # convert to DataFrame
        summary_df = pd.DataFrame(summaries)
        # set to index by KEY
        summary_df.set_index([SurveyCTOPlatform.ID_FIELD], inplace=True)
        summary_df = summary_df.sort_index()

        # set missing field-specific cells to 0
        for col in summary_df.columns:
            if col.startswith("ta_field_"):
                summary_df[col].fillna(0, inplace=True)

        # ensure that DataFrame columns have appropriate data types
        summary_df = summary_df.convert_dtypes()

        return summary_df
