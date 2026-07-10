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

                # paste your existing
                # build_analytics_data()
                # logic here

            elif os.path.exists(
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
                "days": []
            })

        except Exception as e:

            print(
                "analytics error:",
                e
            )

            return jsonify({
                "days": []
            })


analytics_service = AnalyticsService()