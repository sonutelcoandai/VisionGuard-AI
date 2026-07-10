import os
from datetime import datetime

import pandas as pd

from flask import jsonify

from app.core.config import (
    LOG_FILE,
    SAMPLE_FILE
)


class AnalyticsService:

    def _build_analytics_from_sample(
        self,
        sample_df
    ):

        sample_df = sample_df.copy()

        if 'date' not in sample_df.columns:
            sample_df['date'] = (
                datetime.now()
                .date()
                .isoformat()
            )

        if (
            'hour' not in sample_df.columns
            and
            'timestamp' in sample_df.columns
        ):
            sample_df['hour'] = (
                pd.to_datetime(
                    sample_df['timestamp'],
                    errors='coerce'
                )
                .dt.strftime('%H:00')
            )

        if 'visitors' not in sample_df.columns:
            sample_df['visitors'] = 1

        sample_df['hour'] = (
            sample_df['hour']
            .astype(str)
            .str.slice(0, 5)
        )

        days = sorted(
            sample_df['date']
            .dropna()
            .unique()
        )

        hours = [
            f"{str(h).zfill(2)}:00"
            for h in range(9, 22)
        ]

        visitors_per_day = {}
        avg_wait_per_day = {}
        zones_per_day = {}
        gender_counts_per_day = {}
        age_buckets_per_day = {}
        age_gender_map = {}

        for d in days:

            day_df = sample_df[
                sample_df['date'] == d
            ]

            visitors_per_day[d] = [
                int(
                    day_df[
                        day_df['hour'] == h
                    ]['visitors'].sum()
                )
                for h in hours
            ]

            if 'wait_time' in day_df.columns:

                avg_wait_per_day[d] = [
                    float(
                        day_df[
                            day_df['hour'] == h
                        ]['wait_time'].mean()
                        or 0
                    )
                    for h in hours
                ]

            else:

                avg_wait_per_day[d] = [
                    0 for _ in hours
                ]

            if 'zone' in day_df.columns:

                zones_per_day[d] = (
                    day_df['zone']
                    .fillna('')
                    .replace(
                        '',
                        'Cloth Section'
                    )
                    .value_counts()
                    .to_dict()
                )

            else:

                zones_per_day[d] = {}

            if 'gender' in day_df.columns:

                gender_counts_per_day[d] = (
                    day_df['gender']
                    .fillna('Unknown')
                    .replace('', 'Unknown')
                    .value_counts()
                    .to_dict()
                )

            else:

                gender_counts_per_day[d] = {}

            if (
                'age_bucket' in day_df.columns
                and
                'gender' in day_df.columns
            ):

                ag = (
                    day_df
                    .groupby(
                        ['gender', 'age_bucket']
                    )
                    .size()
                    .unstack(fill_value=0)
                )

                age_buckets_per_day[d] = (
                    ag.sum(axis=0).to_dict()
                )

                age_gender_map[d] = {
                    'age_labels':
                        list(ag.columns)
                }

                for g in ag.index:

                    age_gender_map[d][g] = (
                        ag.loc[g]
                        .to_dict()
                    )

            else:

                age_buckets_per_day[d] = {}

                age_gender_map[d] = None

        return {
            "days": days,
            "hours": hours,
            "visitors_per_day": visitors_per_day,
            "avg_wait_per_hour": avg_wait_per_day,
            "zones_per_day": zones_per_day,
            "gender_counts_per_day": gender_counts_per_day,
            "age_buckets_per_day": age_buckets_per_day,
            "age_gender_map": age_gender_map
        }

    def build(self):

        try:

            if (
                os.path.exists(LOG_FILE)
                and
                os.path.getsize(LOG_FILE) > 0
            ):

                df = pd.read_csv(LOG_FILE)

                if "timestamp" in df.columns:

                    df["timestamp"] = pd.to_datetime(
                        df["timestamp"],
                        errors="coerce"
                    )

                    df["date"] = (
                        df["timestamp"]
                        .dt.date
                        .astype(str)
                    )

                    df["hour"] = (
                        df["timestamp"]
                        .dt.strftime("%H:00")
                    )

                else:

                    df["date"] = ""

                    df["hour"] = ""

                hours = [
                    f"{str(h).zfill(2)}:00"
                    for h in range(9, 22)
                ]

                days = sorted(
                    df["date"]
                    .dropna()
                    .unique()
                )

                visitors_per_day = {}

                gender_counts_per_day = {}

                zones_per_day = {}

                age_buckets_per_day = {}

                age_gender_map = {}

                avg_wait_per_hour = {}

                for d in days:

                    day_df = df[
                        df["date"] == d
                    ]

                    visitors_per_day[d] = [
                        int(
                            day_df[
                                day_df["hour"] == h
                            ].shape[0]
                        )
                        for h in hours
                    ]

                    zcounts = (
                        day_df["zone"]
                        .fillna("")
                        .replace(
                            "",
                            "Cloth Section"
                        )
                        .apply(
                            lambda x:
                            (
                                x
                                if x != "unknown"
                                else "Cloth Section"
                            )
                        )
                    )

                    zones_per_day[d] = (
                        zcounts
                        .value_counts()
                        .to_dict()
                    )

                    gender_counts_per_day[d] = (
                        day_df["gender"]
                        .fillna("Unknown")
                        .replace(
                            "",
                            "Unknown"
                        )
                        .value_counts()
                        .to_dict()
                    )

                    if (
                        "age_bucket"
                        in day_df.columns
                    ):

                        age_buckets_per_day[d] = (
                            day_df["age_bucket"]
                            .fillna("Unknown")
                            .replace(
                                "",
                                "Unknown"
                            )
                            .value_counts()
                            .to_dict()
                        )

                    else:

                        age_buckets_per_day[d] = {}

                    if (
                        "age_bucket"
                        in day_df.columns
                        and
                        "gender"
                        in day_df.columns
                    ):

                        ag = (
                            day_df
                            .groupby(
                                [
                                    "gender",
                                    "age_bucket"
                                ]
                            )
                            .size()
                            .unstack(fill_value=0)
                        )

                        age_gender_map[d] = {
                            "age_labels":
                                list(
                                    ag.columns
                                )
                        }

                        for g in ag.index:

                            age_gender_map[d][g] = (
                                ag.loc[g]
                                .to_dict()
                            )

                    else:

                        age_gender_map[d] = None

                    if (
                        "wait_time"
                        in day_df.columns
                    ):

                        avg_wait_per_hour[d] = [
                            float(
                                day_df[
                                    day_df["hour"] == h
                                ][
                                    "wait_time"
                                ].mean()
                                or 0
                            )
                            for h in hours
                        ]

                    else:

                        avg_wait_per_hour[d] = [
                            0 for _ in hours
                        ]

                return jsonify({

                    "days":
                        list(days),

                    "hours":
                        hours,

                    "visitors_per_day":
                        visitors_per_day,

                    "gender_counts_per_day":
                        gender_counts_per_day,

                    "zones_per_day":
                        zones_per_day,

                    "age_buckets_per_day":
                        age_buckets_per_day,

                    "age_gender_map":
                        age_gender_map,

                    "avg_wait_per_hour":
                        avg_wait_per_hour
                })

            if os.path.exists(
                SAMPLE_FILE
            ):

                sample_df = pd.read_csv(
                    SAMPLE_FILE
                )

                return jsonify(
                    self._build_analytics_from_sample(
                        sample_df
                    )
                )

            return jsonify({

                "days": [],
                "hours": [],
                "visitors_per_day": {},
                "gender_counts_per_day": {},
                "zones_per_day": {},
                "age_buckets_per_day": {},
                "age_gender_map": {},
                "avg_wait_per_hour": {}

            })

        except Exception as e:

            print(
                "analytics_data error:",
                e
            )

            return jsonify({

                "days": [],
                "hours": [],
                "visitors_per_day": {},
                "gender_counts_per_day": {},
                "zones_per_day": {},
                "age_buckets_per_day": {},
                "age_gender_map": {},
                "avg_wait_per_hour": {}

            })


analytics_service = AnalyticsService()