"""Deterministic tools for the GB Power Imbalance Risk Agent."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from time import sleep
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd


BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
LONDON = ZoneInfo("Europe/London")
CUTOFF_TIME = time(16, 0)
USER_AGENT = "gb-power-risk-agent/0.4 educational-project"

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_FILE = PROJECT_DIR / "power_risk_model.joblib"
METRICS_FILE = PROJECT_DIR / "model_metrics.csv"

RAW_FEATURES = ["demand", "generation", "imbalance", "margin"]
PUBLICATION_COLUMNS = [
    "demand_published_at",
    "wind_published_at",
    "imbalance_published_at",
    "margin_published_at",
]
SOURCE_START_COLUMNS = [
    "demand_start_time",
    "imbalance_start_time",
    "margin_start_time",
]
AGE_COLUMNS = [
    "demand_age_hours",
    "wind_age_hours",
    "imbalance_age_hours",
    "margin_age_hours",
]
KEYS = ["settlementDate", "settlementPeriod"]

SCENARIOS = {
    "demand_up_5pct": {"demand": 1.05},
    "wind_down_20pct": {"generation": 0.80},
    "margin_down_10pct": {"margin": 0.90},
}

_STATE = {"validations": {}, "predictions": {}}


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("target_date must use YYYY-MM-DD") from error


def _utc_text(value) -> str:
    return pd.Timestamp(value).isoformat().replace("+00:00", "Z")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def _result_id(prefix: str, payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def prediction_cutoff(target_date: date) -> pd.Timestamp:
    previous_day = target_date - timedelta(days=1)
    local_cutoff = datetime.combine(previous_day, CUTOFF_TIME, LONDON)
    return pd.Timestamp(local_cutoff).tz_convert("UTC")


def expected_schedule(target_date: date) -> pd.DataFrame:
    start_local = datetime.combine(target_date, time.min, LONDON)
    end_local = datetime.combine(
        target_date + timedelta(days=1), time.min, LONDON
    )
    starts = pd.date_range(
        pd.Timestamp(start_local).tz_convert("UTC"),
        pd.Timestamp(end_local).tz_convert("UTC"),
        freq="30min",
        inclusive="left",
    )
    return pd.DataFrame({
        "settlementDate": target_date.isoformat(),
        "settlementPeriod": range(1, len(starts) + 1),
        "startTime": starts,
    })


def _get_json(url: str, params: dict | None = None, timeout: int = 180):
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(3):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            sleep(2**attempt)


def _fetch_forecast_stream(
    dataset: str,
    target_date: date,
    boundary: str | None = None,
) -> pd.DataFrame:
    cutoff = prediction_cutoff(target_date)
    params = {
        "publishDateTimeFrom": _utc_text(cutoff - pd.Timedelta("48h")),
        "publishDateTimeTo": _utc_text(cutoff),
    }
    if boundary is not None:
        params["boundary"] = boundary

    payload = _get_json(
        f"{BASE_URL}/datasets/{dataset}/stream", params=params
    )
    frame = pd.DataFrame(payload)
    if frame.empty:
        raise ValueError(f"No {dataset} rows returned")

    frame["publishTime"] = pd.to_datetime(frame["publishTime"], utc=True)
    frame["startTime"] = pd.to_datetime(frame["startTime"], utc=True)
    return frame


def _prepare_keyed_forecast(
    raw: pd.DataFrame,
    target_date: date,
    value_column: str,
    short_name: str,
) -> pd.DataFrame:
    date_text = target_date.isoformat()
    cutoff = prediction_cutoff(target_date)
    data = raw.copy()
    data["settlementDate"] = data["settlementDate"].astype(str)
    data = data[data["settlementDate"] == date_text]
    data = data[data["publishTime"] <= cutoff]
    data["settlementPeriod"] = pd.to_numeric(
        data["settlementPeriod"]
    ).astype("Int64")
    data = data.sort_values(KEYS + ["publishTime"])
    data = data.drop_duplicates(KEYS, keep="last")
    return data[KEYS + ["startTime", "publishTime", value_column]].rename(
        columns={
            "startTime": f"{short_name}_start_time",
            "publishTime": f"{short_name}_published_at",
        }
    )


def _prepare_wind_forecast(
    raw: pd.DataFrame, target_date: date
) -> pd.DataFrame:
    date_text = target_date.isoformat()
    cutoff = prediction_cutoff(target_date)
    wind = raw.copy()
    wind["settlementDate"] = (
        wind["startTime"].dt.tz_convert(LONDON).dt.strftime("%Y-%m-%d")
    )
    wind = wind[wind["settlementDate"] == date_text]
    wind = wind[wind["publishTime"] <= cutoff]
    wind = wind.sort_values(["settlementDate", "startTime", "publishTime"])
    wind = wind.drop_duplicates(
        ["settlementDate", "startTime"], keep="last"
    )
    return wind[[
        "settlementDate", "startTime", "publishTime", "generation"
    ]].rename(columns={"publishTime": "wind_published_at"})


def _fetch_live_day(date_text: str) -> pd.DataFrame:
    target_date = _parse_date(date_text)
    schedule = expected_schedule(target_date)

    demand = _prepare_keyed_forecast(
        _fetch_forecast_stream("NDF", target_date, boundary="N"),
        target_date,
        "demand",
        "demand",
    )
    imbalance = _prepare_keyed_forecast(
        _fetch_forecast_stream("IMBALNGC", target_date, boundary="N"),
        target_date,
        "imbalance",
        "imbalance",
    )
    margin = _prepare_keyed_forecast(
        _fetch_forecast_stream("MELNGC", target_date, boundary="N"),
        target_date,
        "margin",
        "margin",
    )
    wind = _prepare_wind_forecast(
        _fetch_forecast_stream("WINDFOR", target_date), target_date
    )

    data = schedule.merge(demand, on=KEYS, how="left")
    data = data.merge(imbalance, on=KEYS, how="left")
    data = data.merge(margin, on=KEYS, how="left")
    data = pd.merge_asof(
        data.sort_values("startTime"),
        wind.sort_values("startTime"),
        on="startTime",
        by="settlementDate",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
    )
    data["prediction_cutoff"] = prediction_cutoff(target_date)
    return _normalise_frame(data, target_date)


def _normalise_frame(frame: pd.DataFrame, target_date: date) -> pd.DataFrame:
    data = frame.copy()
    data["settlementDate"] = data["settlementDate"].astype(str)
    data["settlementPeriod"] = pd.to_numeric(
        data["settlementPeriod"]
    ).astype("Int64")

    timestamp_columns = [
        "startTime",
        "prediction_cutoff",
        *PUBLICATION_COLUMNS,
        *SOURCE_START_COLUMNS,
    ]
    for column in timestamp_columns:
        if column in data:
            data[column] = pd.to_datetime(data[column], utc=True)

    for column in RAW_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    cutoff = prediction_cutoff(target_date)
    data["prediction_cutoff"] = cutoff
    local_start = data["startTime"].dt.tz_convert(LONDON)
    data["local_hour"] = local_start.dt.hour + local_start.dt.minute / 60
    data["day_of_week"] = local_start.dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)

    publication_map = {
        "demand": "demand_published_at",
        "wind": "wind_published_at",
        "imbalance": "imbalance_published_at",
        "margin": "margin_published_at",
    }
    for name, column in publication_map.items():
        data[f"{name}_age_hours"] = (
            cutoff - data[column]
        ).dt.total_seconds() / 3600

    return data.sort_values(KEYS).reset_index(drop=True)


def construct_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Recreate the exact 15 features used in notebook 03."""
    data = frame.copy()
    starts = pd.to_datetime(data["startTime"], utc=True)
    local_starts = starts.dt.tz_convert(LONDON)
    dates = pd.to_datetime(data["settlementDate"])

    data["local_hour"] = (
        local_starts.dt.hour + local_starts.dt.minute / 60
    )
    data["day_of_week"] = local_starts.dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["residual_demand"] = data["demand"] - data["generation"]
    data["wind_share"] = data["generation"] / data["demand"]
    data["margin_ratio"] = data["margin"] / data["demand"]
    data["imbalance_ratio"] = data["imbalance"] / data["demand"]
    data["hour_sin"] = np.sin(2 * np.pi * data["local_hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["local_hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    data["month_sin"] = np.sin(
        2 * np.pi * (dates.dt.month - 1) / 12
    )
    data["month_cos"] = np.cos(
        2 * np.pi * (dates.dt.month - 1) / 12
    )
    return data


@lru_cache(maxsize=1)
def _load_bundle() -> dict:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("power_risk_model.joblib is missing")
    bundle = joblib.load(MODEL_FILE)
    required = {
        "champion_name",
        "champion_model",
        "features",
        "trained_through",
        "training_rows",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle is missing: {sorted(missing)}")
    bundle["model_id"] = f"model_{_hash_file(MODEL_FILE)}"
    return bundle


def _data_id(frame: pd.DataFrame) -> str:
    columns = KEYS + RAW_FEATURES + PUBLICATION_COLUMNS
    text = frame[columns].to_csv(index=False, float_format="%.8g")
    return f"data_{hashlib.sha256(text.encode()).hexdigest()[:12]}"


def _validate_frame(frame: pd.DataFrame, target_date: date) -> dict:
    schedule = expected_schedule(target_date)
    cutoff = prediction_cutoff(target_date)
    expected_periods = len(schedule)
    expected_sequence = list(range(1, expected_periods + 1))

    checks = {
        "duplicate_keys": int(frame.duplicated(KEYS).sum()),
        "missing_raw_values": int(frame[RAW_FEATURES].isna().sum().sum()),
        "missing_publication_times": int(
            frame[PUBLICATION_COLUMNS].isna().sum().sum()
        ),
        "late_forecasts": int(
            sum((frame[column] > cutoff).sum() for column in PUBLICATION_COLUMNS)
        ),
        "stale_forecasts": int(
            sum((frame[column] > 24).sum() for column in AGE_COLUMNS)
        ),
        "source_time_mismatches": 0,
        "schedule_mismatches": 0,
        "non_finite_features": 0,
    }

    for column in SOURCE_START_COLUMNS:
        valid = frame[column].notna()
        checks["source_time_mismatches"] += int(
            (frame.loc[valid, column] != frame.loc[valid, "startTime"]).sum()
        )

    schedule_check = schedule.merge(
        frame[KEYS + ["startTime"]], on=KEYS, how="left", suffixes=("_expected", "_actual")
    )
    checks["schedule_mismatches"] = int(
        (
            schedule_check["startTime_expected"]
            != schedule_check["startTime_actual"]
        ).sum()
    )

    impossible_values = int((frame["demand"] <= 0).sum())
    impossible_values += int((frame["generation"] < 0).sum())
    checks["impossible_values"] = impossible_values

    issues = []
    if len(frame) != expected_periods:
        issues.append("Incorrect number of settlement periods")
    if frame["settlementPeriod"].astype(int).tolist() != expected_sequence:
        issues.append("Settlement-period sequence is broken")
    if not (frame["settlementDate"] == target_date.isoformat()).all():
        issues.append("Rows from another settlement date are present")

    for check_name in [
        "duplicate_keys",
        "missing_raw_values",
        "missing_publication_times",
        "late_forecasts",
        "stale_forecasts",
        "source_time_mismatches",
        "schedule_mismatches",
        "impossible_values",
    ]:
        if checks[check_name] > 0:
            issues.append(check_name.replace("_", " ").capitalize())

    bundle = _load_bundle()
    feature_frame = construct_model_features(frame)
    feature_columns = bundle["features"]
    matrix = feature_frame[feature_columns]
    checks["non_finite_features"] = int(
        (~np.isfinite(matrix.to_numpy(dtype=float))).sum()
    )
    if checks["non_finite_features"]:
        issues.append("Constructed model features are not finite")

    champion_columns = list(
        getattr(bundle["champion_model"], "feature_names_in_", feature_columns)
    )
    if champion_columns != feature_columns:
        issues.append("Saved model feature order does not match the bundle")

    age_summary = {
        column.replace("_age_hours", "_max_hours"): round(
            float(frame[column].max()), 2
        )
        for column in AGE_COLUMNS
    }
    return {
        "checks": checks,
        "critical_issues": issues,
        "expected_periods": expected_periods,
        "publication_age": age_summary,
    }


def check_data(target_date: str) -> dict:
    """Check Elexon forecasts and the saved model before any prediction.

    Args:
        target_date: Settlement date in YYYY-MM-DD format.

    Returns:
        A validation result. Prediction must stop when can_predict is false.
    """
    try:
        parsed_date = _parse_date(target_date)
        cutoff = prediction_cutoff(parsed_date)
        if pd.Timestamp.now(tz="UTC") < cutoff:
            return {
                "tool": "check_data",
                "status": "ABSTAIN",
                "can_predict": False,
                "target_date": target_date,
                "cutoff_utc": cutoff.isoformat(),
                "critical_issues": [
                    "The D-1 16:00 Europe/London cutoff has not passed"
                ],
            }

        frame = _fetch_live_day(target_date).copy()
        validation = _validate_frame(frame, parsed_date)
        data_id = _data_id(frame)
        model_id = _load_bundle()["model_id"]
        can_predict = not validation["critical_issues"]
        status = "PASS" if can_predict else "ABSTAIN"
        payload = {
            "target_date": target_date,
            "data_id": data_id,
            "model_id": model_id,
            "checks": validation["checks"],
        }
        validation_id = _result_id("validation", payload)
        result = {
            "tool": "check_data",
            "status": status,
            "can_predict": can_predict,
            "validation_id": validation_id,
            "data_id": data_id,
            "model_id": model_id,
            "target_date": target_date,
            "cutoff_utc": cutoff.isoformat(),
            "expected_periods": validation["expected_periods"],
            "observed_periods": len(frame),
            "checks": validation["checks"],
            "publication_age": validation["publication_age"],
            "critical_issues": validation["critical_issues"],
            "deployment_note": (
                "Post-test live use" if parsed_date.year > 2025
                else "Historical replay, not a new out-of-sample forecast"
            ),
        }
        _STATE["validations"][validation_id] = {
            "result": result,
            "frame": frame,
        }
        return result
    except Exception as error:
        return {
            "tool": "check_data",
            "status": "ABSTAIN",
            "can_predict": False,
            "target_date": target_date,
            "critical_issues": [f"Validation failed: {error}"],
        }


def _get_validation(target_date: str, validation_id: str):
    record = _STATE["validations"].get(validation_id)
    if record is None:
        raise ValueError("Unknown validation_id. Call check_data first")
    if record["result"]["target_date"] != target_date:
        raise ValueError("validation_id belongs to another target date")
    if not record["result"]["can_predict"]:
        raise ValueError("The data check did not pass")
    return record


def _prediction_frame(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    data = construct_model_features(frame)
    feature_columns = bundle["features"]
    matrix = data[feature_columns]
    if list(matrix.columns) != feature_columns:
        raise ValueError("Model feature order changed")
    data["champion_probability"] = bundle[
        "champion_model"
    ].predict_proba(matrix)[:, 1]
    data["daily_risk_rank"] = data["champion_probability"].rank(
        method="first", ascending=False
    ).astype(int)
    return data


def _period_record(row: pd.Series, daily_medians: pd.Series) -> dict:
    local_time = pd.Timestamp(row["startTime"]).tz_convert(LONDON)
    return {
        "period_id": f"{row['settlementDate']}_SP{int(row['settlementPeriod'])}",
        "settlement_period": int(row["settlementPeriod"]),
        "start_time_local": local_time.strftime("%Y-%m-%d %H:%M %Z"),
        "champion_probability_pct": round(
            float(row["champion_probability"]) * 100, 1
        ),
        "daily_risk_rank": int(row["daily_risk_rank"]),
        "forecast_context_mw": {
            variable: {
                "value": round(float(row[variable]), 1),
                "daily_median": round(float(daily_medians[variable]), 1),
            }
            for variable in RAW_FEATURES
        },
    }


def predict_risk(target_date: str, validation_id: str) -> dict:
    """Predict short-system probabilities after a passed data check.

    Args:
        target_date: Settlement date in YYYY-MM-DD format.
        validation_id: Identifier returned by check_data for the same date.

    Returns:
        Model summary and the six highest-risk settlement periods.
    """
    try:
        record = _get_validation(target_date, validation_id)
        bundle = _load_bundle()
        predictions = _prediction_frame(record["frame"], bundle)
        top_periods = predictions.nsmallest(6, "daily_risk_rank")
        daily_medians = predictions[RAW_FEATURES].median()
        summary = {
            "mean_short_probability_pct": round(
                float(predictions["champion_probability"].mean()) * 100, 1
            ),
            "maximum_short_probability_pct": round(
                float(predictions["champion_probability"].max()) * 100, 1
            ),
            "minimum_short_probability_pct": round(
                float(predictions["champion_probability"].min()) * 100, 1
            ),
        }
        period_predictions = [
            {
                "settlement_period": int(row["settlementPeriod"]),
                "start_time_local": (
                    pd.Timestamp(row["startTime"])
                    .tz_convert(LONDON)
                    .strftime("%Y-%m-%d %H:%M %Z")
                ),
                "short_probability_pct": round(
                    float(row["champion_probability"]) * 100,
                    1,
                ),
                "daily_risk_rank": int(row["daily_risk_rank"]),
            }
            for _, row in predictions.sort_values(
                "settlementPeriod"
            ).iterrows()
        ]        
        payload = {
            "validation_id": validation_id,
            "model_id": record["result"]["model_id"],
            "summary": summary,
            }
        prediction_id = _result_id("prediction", payload)
        result = {
            "tool": "predict_risk",
            "status": "READY_FOR_HUMAN_REVIEW",
            "prediction_id": prediction_id,
            "validation_id": validation_id,
            "data_id": record["result"]["data_id"],
            "model_id": record["result"]["model_id"],
            "target_date": target_date,
            "model": bundle["champion_name"],
            "trained_through": bundle["trained_through"],
            "period_count": len(predictions),
            "summary": summary,
            "period_predictions": period_predictions,            
            "highest_risk_periods": [
                _period_record(row, daily_medians)
                for _, row in top_periods.iterrows()
            ],
            "deployment_note": record["result"]["deployment_note"],
            "decision_note": "Decision support only. No trading action is produced.",
        }
        _STATE["predictions"][prediction_id] = {
            "result": result,
            "frame": predictions,
            "base_frame": record["frame"].copy(),
        }
        return result
    except Exception as error:
        return {
            "tool": "predict_risk",
            "status": "ABSTAIN",
            "target_date": target_date,
            "reason": str(error),
        }

def run_scenario(
    prediction_id: str,
    scenario: str = "custom",
    demand_change_pct: float = 0.0,
    wind_change_pct: float = 0.0,
    margin_change_mw: float = 0.0,
) -> dict:
    """Run one bounded whole-day model sensitivity."""
    try:
        record = _STATE["predictions"].get(prediction_id)

        if record is None:
            raise ValueError(
                "Unknown prediction_id. Call predict_risk first"
            )

        if scenario != "custom":
            raise ValueError("scenario must be custom")

        demand_change_pct = float(demand_change_pct)
        wind_change_pct = float(wind_change_pct)
        margin_change_mw = float(margin_change_mw)

        if not -10 <= demand_change_pct <= 10:
            raise ValueError(
                "demand_change_pct must be between -10 and 10"
            )

        if not -30 <= wind_change_pct <= 30:
            raise ValueError(
                "wind_change_pct must be between -30 and 30"
            )

        if not -2000 <= margin_change_mw <= 2000:
            raise ValueError(
                "margin_change_mw must be between -2000 and 2000"
            )

        bundle = _load_bundle()
        scenario_frame = record["base_frame"].copy()

        assumptions = {
            "demand_change_pct": demand_change_pct,
            "wind_change_pct": wind_change_pct,
            "margin_change_mw": margin_change_mw,
        }

        scenario_frame["demand"] *= (
            1 + demand_change_pct / 100
        )
        scenario_frame["generation"] *= (
            1 + wind_change_pct / 100
        )
        scenario_frame["margin"] += margin_change_mw

        if (scenario_frame["demand"] <= 0).any():
            raise ValueError(
                "Scenario produces non-positive demand"
            )

        if (scenario_frame["generation"] < 0).any():
            raise ValueError(
                "Scenario produces negative wind generation"
            )

        features = construct_model_features(scenario_frame)

        scenario_probability = (
            bundle["champion_model"]
            .predict_proba(features[bundle["features"]])[:, 1]
        )

        comparison = record["frame"][
            KEYS + ["champion_probability"]
        ].copy()

        comparison["scenario_probability"] = (
            scenario_probability
        )

        comparison["change_pp"] = (
            comparison["scenario_probability"]
            - comparison["champion_probability"]
        ) * 100

        most_changed = comparison.reindex(
            comparison["change_pp"]
            .abs()
            .sort_values(ascending=False)
            .index
        ).head(6)

        period_results = [
            {
                "settlement_period": int(
                    row["settlementPeriod"]
                ),
                "baseline_probability_pct": round(
                    float(row["champion_probability"]) * 100,
                    1,
                ),
                "scenario_probability_pct": round(
                    float(row["scenario_probability"]) * 100,
                    1,
                ),
                "change_pp": round(
                    float(row["change_pp"]),
                    1,
                ),
            }
            for _, row in comparison.sort_values(
                "settlementPeriod"
            ).iterrows()
        ]

        most_changed_periods = [
            {
                "settlement_period": int(
                    row["settlementPeriod"]
                ),
                "baseline_probability_pct": round(
                    float(row["champion_probability"]) * 100,
                    1,
                ),
                "scenario_probability_pct": round(
                    float(row["scenario_probability"]) * 100,
                    1,
                ),
                "change_pp": round(
                    float(row["change_pp"]),
                    1,
                ),
            }
            for _, row in most_changed.iterrows()
        ]

        return {
            "tool": "run_scenario",
            "status": "READY_FOR_HUMAN_REVIEW",
            "result_id": _result_id(
                "scenario",
                {
                    "prediction_id": prediction_id,
                    "scenario": scenario,
                    "assumptions": assumptions,
                },
            ),
            "prediction_id": prediction_id,
            "scenario": scenario,
            "assumptions": assumptions,
            "summary": {
                "baseline_mean_probability_pct": round(
                    float(
                        comparison[
                            "champion_probability"
                        ].mean()
                    ) * 100,
                    1,
                ),
                "scenario_mean_probability_pct": round(
                    float(
                        comparison[
                            "scenario_probability"
                        ].mean()
                    ) * 100,
                    1,
                ),
                "mean_change_pp": round(
                    float(comparison["change_pp"].mean()),
                    1,
                ),
                "maximum_absolute_change_pp": round(
                    float(
                        comparison["change_pp"].abs().max()
                    ),
                    1,
                ),
            },
            "period_results": period_results,
            "most_changed_periods": most_changed_periods,
            "method_note": (
                "This is a bounded model sensitivity with "
                "all other forecasts held constant. "
                "It is not a new market forecast."
            ),
        }

    except Exception as error:
        return {
            "tool": "run_scenario",
            "status": "ABSTAIN",
            "prediction_id": prediction_id,
            "scenario": scenario,
            "reason": str(error),
        }

def compare_with_actuals(prediction_id: str) -> dict:
    """Compare predicted probabilities with realised Elexon outcomes."""

    try:
        record = _STATE["predictions"].get(prediction_id)

        if record is None:
            raise ValueError(
                "Unknown prediction_id. Call predict_risk first"
            )

        target_date = record["result"]["target_date"]
        parsed_date = _parse_date(target_date)

        if parsed_date >= datetime.now(LONDON).date():
            raise ValueError(
                "Actual outcomes are only available for completed dates"
            )

        url = (
            f"{BASE_URL}/balancing/settlement/"
            f"system-prices/{target_date}"
        )

        payload = _get_json(url, timeout=60)
        actuals = pd.DataFrame(payload.get("data", []))

        required_columns = {
            "settlementPeriod",
            "startTime",
            "netImbalanceVolume",
        }

        if actuals.empty:
            raise ValueError("No realised outcomes were returned")

        if not required_columns.issubset(actuals.columns):
            raise ValueError("The realised outcome data are incomplete")

        actuals["settlementDate"] = target_date

        actuals["settlementPeriod"] = pd.to_numeric(
            actuals["settlementPeriod"],
            errors="coerce",
        ).astype("Int64")

        actuals["netImbalanceVolume"] = pd.to_numeric(
            actuals["netImbalanceVolume"],
            errors="coerce",
        )

        actuals["outcome_start_time"] = pd.to_datetime(
            actuals["startTime"],
            utc=True,
            errors="coerce",
        )

        if actuals.duplicated(KEYS).any():
            raise ValueError("Duplicate realised settlement periods found")

        predictions = record["frame"][
            KEYS + ["startTime", "champion_probability"]
        ].copy()

        comparison = predictions.merge(
            actuals[
                KEYS
                + ["outcome_start_time", "netImbalanceVolume"]
            ],
            on=KEYS,
            how="left",
            validate="one_to_one",
        )

        missing_outcomes = int(
            comparison["netImbalanceVolume"].isna().sum()
        )

        if missing_outcomes:
            raise ValueError(
                f"Official outcome is missing for "
                f"{missing_outcomes} settlement periods"
            )

        time_mismatches = int(
            (
                comparison["outcome_start_time"]
                != comparison["startTime"]
            ).sum()
        )

        if time_mismatches:
            raise ValueError(
                f"Outcome time disagrees for "
                f"{time_mismatches} settlement periods"
            )

        comparison["actual_system_short"] = (
            comparison["netImbalanceVolume"] > 0
        )

        comparison["predicted_system_short"] = (
            comparison["champion_probability"] >= 0.50
        )

        comparison["correct_at_50pct"] = (
            comparison["actual_system_short"]
            == comparison["predicted_system_short"]
        )

        actual = comparison["actual_system_short"].astype(float)
        probability = comparison["champion_probability"].astype(float)

        brier_score = float(
            ((probability - actual) ** 2).mean()
        )

        summary = {
            "mean_predicted_probability_pct": round(
                float(probability.mean()) * 100,
                1,
            ),
            "actual_short_share_pct": round(
                float(actual.mean()) * 100,
                1,
            ),
            "accuracy_at_50pct_pct": round(
                float(
                    comparison["correct_at_50pct"].mean()
                ) * 100,
                1,
            ),
            "daily_brier_score": round(brier_score, 4),
        }

        period_results = []

        for _, row in comparison.iterrows():
            local_time = pd.Timestamp(
                row["startTime"]
            ).tz_convert(LONDON)

            period_results.append({
                "settlement_period": int(
                    row["settlementPeriod"]
                ),
                "start_time_local": local_time.strftime(
                    "%Y-%m-%d %H:%M %Z"
                ),
                "predicted_probability_pct": round(
                    float(row["champion_probability"]) * 100,
                    1,
                ),
                "actual_system_short": bool(
                    row["actual_system_short"]
                ),
                "realised_niv_mw": round(
                    float(row["netImbalanceVolume"]),
                    1,
                ),
                "correct_at_50pct": bool(
                    row["correct_at_50pct"]
                ),
            })

        result = {
            "tool": "compare_with_actuals",
            "status": "READY_FOR_HUMAN_REVIEW",
            "result_id": _result_id(
                "comparison",
                {
                    "prediction_id": prediction_id,
                    "summary": summary,
                },
            ),
            "prediction_id": prediction_id,
            "target_date": target_date,
            "period_count": len(comparison),
            "summary": summary,
            "period_results": period_results,
            "comparison_note": (
                "Realised NIV is shown after settlement. "
                "The model predicted short-system probability, "
                "not NIV magnitude."
            ),
            "metric_note": (
                "These metrics describe one completed day. "
                "They do not replace the full 2025 evaluation."
            ),
        }

        return result

    except Exception as error:
        return {
            "tool": "compare_with_actuals",
            "status": "ABSTAIN",
            "prediction_id": prediction_id,
            "reason": str(error),
        }

def get_model_card() -> dict:
    """Return the frozen model design, test results, and limitations."""
    try:
        bundle = _load_bundle()
        test_metrics = {
            key: round(float(value), 6)
            for key, value in bundle.get("test_metrics", {}).items()
            if isinstance(value, (int, float, np.number))
        }
        baseline_brier = None
        if METRICS_FILE.exists():
            metrics = pd.read_csv(METRICS_FILE)
            baseline = metrics[
                (metrics["stage"] == "2025 test")
                & (metrics["model"] == "Constant baseline")
            ]
            if len(baseline) == 1:
                baseline_brier = round(float(baseline.iloc[0]["brier"]), 6)

        result = {
            "tool": "get_model_card",
            "status": "READY_FOR_HUMAN_REVIEW",
            "model_id": bundle["model_id"],
            "champion": bundle["champion_name"],
            "benchmark": "Logistic regression",
            "development_period": "2022 to 2024",
            "selection_period": "2024",
            "evaluation_period": "2025",
            "production_refit_through": bundle["trained_through"],
            "training_rows": int(bundle["training_rows"]),
            "cutoff_policy": bundle.get("cutoff_policy"),
            "test_metrics": test_metrics,
            "constant_test_brier": baseline_brier,
            "features": bundle["features"],
            "limitations": [
                "Moderate discrimination, not a trading strategy",
                "Half-hours within a day are related",
                "Live dates after 2025 are post-test use",
                "Scenario sensitivities are not causal effects",
            ],
        }
        result["result_id"] = _result_id("model_card", result)
        return result
    except Exception as error:
        return {
            "tool": "get_model_card",
            "status": "ABSTAIN",
            "reason": str(error),
        }


TOOLS = {
    "check_data": check_data,
    "predict_risk": predict_risk,
    "run_scenario": run_scenario,
    "compare_with_actuals": compare_with_actuals,
    "get_model_card": get_model_card,
}
