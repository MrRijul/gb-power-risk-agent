"""Streamlit interface for the GB Power Imbalance Risk Agent."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from agent import run_agent
from tools import (
    check_data,
    compare_with_actuals,
    get_model_card,
    predict_risk,
    run_scenario,
)


LONDON = ZoneInfo("Europe/London")


def clear_results() -> None:
    """Remove results that belong to an earlier date or validation."""
    for key in ["validation", "prediction", "actuals", "scenario"]:
        st.session_state.pop(key, None)


def render_validation(result: dict) -> None:
    if result["status"] == "ABSTAIN":
        st.error("Data status: ABSTAIN")
        for issue in result.get("critical_issues", []):
            st.write(f"- {issue}")
        if result.get("cutoff_utc"):
            st.caption(f"Required cutoff: {result['cutoff_utc']}")
        return

    st.success("Data status: PASS")
    first, second, third = st.columns(3)
    first.metric(
        "Settlement periods",
        f"{result['observed_periods']} / {result['expected_periods']}",
    )
    second.metric("Late forecasts", result["checks"]["late_forecasts"])
    third.metric("Stale forecasts", result["checks"]["stale_forecasts"])
    st.caption(f"Cutoff: {result['cutoff_utc']}")
    st.caption(f"Evidence: {result['validation_id']}")


def render_prediction(result: dict) -> None:
    if result["status"] == "ABSTAIN":
        st.error(f"Prediction stopped: {result['reason']}")
        return

    summary = result["summary"]
    first, second, third = st.columns(3)
    first.metric("Mean probability", f"{summary['mean_short_probability_pct']}%")
    second.metric(
        "Maximum probability", f"{summary['maximum_short_probability_pct']}%"
    )
    third.metric(
        "Minimum probability", f"{summary['minimum_short_probability_pct']}%"
    )

    period_rows = result.get("period_predictions")

    if not period_rows:
        st.warning(
            "This prediction came from an older tool version. "
            "Run the data check and generate the assessment again."
        )
        return

    profile = pd.DataFrame(period_rows).rename(
        columns={
            "settlement_period": "Settlement period",
            "start_time_local": "London time",
            "short_probability_pct": "Short probability (%)",
        }
    )

    profile["Forecast interval"] = (
        "SP "
        + profile["Settlement period"].astype(str).str.zfill(2)
        + " | "
        + profile["London time"].str.slice(11)
    )

    st.line_chart(
        profile.set_index("Forecast interval")[
            ["Short probability (%)"]
        ]
    )

    st.caption(
        "Each point shows the probability that the GB system will be "
        "short during that London settlement period."
    )    

    risk_rows = []
    for period in result["highest_risk_periods"]:
        context = period["forecast_context_mw"]
        risk_rows.append({
            "Settlement period": period["settlement_period"],
            "Local time": period["start_time_local"],
            "Short probability (%)": period["champion_probability_pct"],
            "Demand (MW)": context["demand"]["value"],
            "Wind (MW)": context["generation"]["value"],
            "Indicated imbalance (MW)": context["imbalance"]["value"],
            "Margin (MW)": context["margin"]["value"],
        })

    st.subheader("Highest-risk settlement periods")
    st.dataframe(
        pd.DataFrame(risk_rows), hide_index=True, width="stretch"
    )
    st.caption(f"Model: {result['model']}")
    st.caption(f"Evidence: {result['prediction_id']}")
    st.info(result["deployment_note"])


def render_actuals(result: dict) -> None:
    if result["status"] == "ABSTAIN":
        st.error(f"Comparison stopped: {result['reason']}")
        return

    summary = result["summary"]
    first, second, third, fourth = st.columns(4)
    first.metric(
        "Mean predicted probability",
        f"{summary['mean_predicted_probability_pct']}%",
    )
    second.metric(
        "Actual short-period share", f"{summary['actual_short_share_pct']}%"
    )
    third.metric(
        "Accuracy at 50%", f"{summary['accuracy_at_50pct_pct']}%"
    )
    fourth.metric("Daily Brier score", summary["daily_brier_score"])

    table = pd.DataFrame(result["period_results"]).rename(
        columns={
            "settlement_period": "Settlement period",
            "start_time_local": "Local time",
            "predicted_probability_pct": "Predicted probability (%)",
            "actual_system_short": "Actual system short",
            "realised_niv_mw": "Realised NIV (MW)",
            "correct_at_50pct": "Correct at 50%",
        }
    )
    table["Actual status"] = table["Actual system short"].map(
        {True: "SHORT", False: "LONG"}
    )

    probability_tab, niv_tab = st.tabs(
        ["Predicted probability", "Realised NIV"]
    )
    with probability_tab:
        st.line_chart(
            table.set_index("Settlement period")[["Predicted probability (%)"]]
        )
    with niv_tab:
        st.bar_chart(
            table.set_index("Settlement period")[["Realised NIV (MW)"]]
        )

    st.dataframe(
        table[
            [
                "Settlement period",
                "Local time",
                "Predicted probability (%)",
                "Actual status",
                "Realised NIV (MW)",
                "Correct at 50%",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Positive realised NIV means the system was short. "
        "Negative realised NIV means it was long."
    )
    st.info(result["comparison_note"])
    st.caption(result["metric_note"])
    st.caption(f"Evidence: {result['result_id']}")


def render_scenario(result: dict) -> None:
    if result["status"] == "ABSTAIN":
        st.error(f"Scenario stopped: {result['reason']}")
        return

    summary = result["summary"]
    assumptions = result["assumptions"]
    st.markdown(
        "**Scenario assumptions:** "
        f"Demand {assumptions.get('demand_change_pct', 0):+g}%, "
        f"wind {assumptions.get('wind_change_pct', 0):+g}%, "
        f"margin {assumptions.get('margin_change_mw', 0):+g} MW"
    )

    first, second, third = st.columns(3)
    first.metric(
        "Baseline mean probability",
        f"{summary['baseline_mean_probability_pct']}%",
    )
    second.metric(
        "Scenario mean probability",
        f"{summary['scenario_mean_probability_pct']}%",
        delta=f"{summary['mean_change_pp']:+.1f} pp",
        delta_color="inverse",
    )
    third.metric(
        "Largest period change",
        f"{summary['maximum_absolute_change_pp']} pp",
    )
       
    profile = pd.DataFrame(result["period_results"]).rename(
        columns={
            "settlement_period": "Settlement period",
            "baseline_probability_pct": "Baseline probability (%)",
            "scenario_probability_pct": "Scenario probability (%)",
            "change_pp": "Change (pp)",
        }
    )
    st.line_chart(
        profile.set_index("Settlement period")[
            ["Baseline probability (%)", "Scenario probability (%)"]
        ]
    )

    affected = pd.DataFrame(result["most_changed_periods"]).rename(
        columns={
            "settlement_period": "Settlement period",
            "baseline_probability_pct": "Baseline probability (%)",
            "scenario_probability_pct": "Scenario probability (%)",
            "change_pp": "Change (pp)",
        }
    )
    st.subheader("Most affected settlement periods")
    st.dataframe(affected, hide_index=True, width="stretch")
    st.info(result["method_note"])
    st.caption(f"Evidence: {result['result_id']}")


def render_agent_trace(audit: dict) -> None:
    rows = [
        {
            "Step": step["sequence"],
            "Tool": step["tool"],
            "Status": step["status"],
            "Arguments": step["arguments"],
            "Evidence": step["result_id"],
            "Runtime (ms)": step["elapsed_ms"],
        }
        for step in audit.get("steps", [])
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_agent_outputs(audit: dict) -> None:
    """Render charts from the structured tool results saved in the audit."""
    successful = {
        step["tool"]: step["result"]
        for step in audit.get("steps", [])
        if step["result"].get("status") != "ABSTAIN"
    }
    if "compare_with_actuals" in successful:
        st.subheader("Visual comparison")
        render_actuals(successful["compare_with_actuals"])
    if "run_scenario" in successful:
        st.subheader("Scenario slide")
        render_scenario(successful["run_scenario"])


st.set_page_config(
    page_title="GB Power Imbalance Risk Agent",
    layout="wide",
)

st.title("GB Power Imbalance Risk Agent")
st.write(
    "Point-in-time probability assessment, realised-outcome review and "
    "bounded scenario analysis for the GB electricity system."
)

model_card = get_model_card()
if model_card["status"] == "ABSTAIN":
    st.error(model_card["reason"])
    st.stop()

with st.sidebar:
    st.subheader("Model")
    st.write(f"Champion: {model_card['champion']}")
    st.write(f"Evaluation period: {model_card['evaluation_period']}")
    st.write(f"Training observations: {model_card['training_rows']:,}")
    metrics = model_card.get("test_metrics", {})
    if "auc" in metrics:
        st.write(f"2025 test AUC: {metrics['auc']:.3f}")
    if "brier" in metrics:
        st.write(f"2025 Brier score: {metrics['brier']:.3f}")

agent_tab, dashboard_tab = st.tabs([
    "Research agent",
    "Risk dashboard",
])

with dashboard_tab:
    selected_date = st.date_input(
        "Settlement date",
        value=datetime.now(LONDON).date(),
        key="dashboard_date",
    )
    target_date = selected_date.isoformat()

    if st.session_state.get("active_date") != target_date:
        st.session_state["active_date"] = target_date
        clear_results()

    if st.button("Check data", type="primary"):
        clear_results()
        with st.spinner("Checking Elexon forecasts and publication times..."):
            st.session_state["validation"] = check_data(target_date)

    validation = st.session_state.get("validation")
    if validation:
        render_validation(validation)

    if validation and validation.get("can_predict"):
        if st.button("Generate risk assessment"):
            st.session_state.pop("prediction", None)
            st.session_state.pop("actuals", None)
            st.session_state.pop("scenario", None)
            with st.spinner("Running the probability model..."):
                st.session_state["prediction"] = predict_risk(
                    target_date=target_date,
                    validation_id=validation["validation_id"],
                )

    prediction = st.session_state.get("prediction")
    if prediction:
        st.divider()
        st.header("Probability assessment")
        render_prediction(prediction)

    if (
        prediction
        and prediction["status"] != "ABSTAIN"
        and selected_date < datetime.now(LONDON).date()
    ):
        st.divider()
        st.header("Forecast versus actual outcome")
        if st.button("Compare with actual outcome"):
            with st.spinner("Downloading realised Elexon outcomes..."):
                st.session_state["actuals"] = compare_with_actuals(
                    prediction["prediction_id"]
                )

        actuals = st.session_state.get("actuals")
        if actuals:
            render_actuals(actuals)

    if prediction and prediction["status"] != "ABSTAIN":
        st.divider()
        st.header("Model sensitivity scenario")
        st.caption(
            "Adjust selected inputs while holding the remaining model inputs "
            "constant."
        )

        first, second, third = st.columns(3)
        demand_change = first.slider(
            "Demand change (%)", -10, 10, 5, 1
        )
        wind_change = second.slider(
            "Wind-generation change (%)", -30, 30, -15, 1
        )
        margin_change = third.slider(
            "Margin change (MW)", -2000, 2000, 0, 100
        )

        inputs = (demand_change, wind_change, margin_change)
        if st.session_state.get("scenario_inputs") != inputs:
            st.session_state["scenario_inputs"] = inputs
            st.session_state.pop("scenario", None)

        if st.button("Run sensitivity"):
            with st.spinner("Recalculating settlement-period probabilities..."):
                st.session_state["scenario"] = run_scenario(
                    prediction_id=prediction["prediction_id"],
                    scenario="custom",
                    demand_change_pct=demand_change,
                    wind_change_pct=wind_change,
                    margin_change_mw=margin_change,
                )

        scenario = st.session_state.get("scenario")
        if scenario:
            render_scenario(scenario)

with agent_tab:
    st.subheader("Research agent")

    st.write("Ask the agent to:")

    st.markdown(
        """
        - Check whether the forecast data is safe to use.
        - Estimate probabilities that system is short.
        - Compare a historical forecast with the realised outcome.
        - Test demand, wind and margin scenarios.
        - Explain the model and its 2025 performance.
        """
    )

    st.caption(
        "The agent chooses the relevant tools, but Python produces every "
        "number."
    )

    st.info(
        "Example: Assess 2026-08-11. What if demand is 7% higher, "
        "wind is 15% lower and margin falls by 800 MW?"
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.text(message["content"])
            if message.get("audit"):
                with st.expander("Tool trace"):
                    render_agent_trace(message["audit"])
                render_agent_outputs(message["audit"])

    question = st.chat_input(
        "Ask about a date, realised outcomes, model quality or a scenario"
    )

    if question:
        st.session_state["chat_history"].append({
            "role": "user",
            "content": question,
        })
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("The agent is selecting and running tools..."):
                try:
                    report, audit = run_agent(question)
                except Exception as error:
                    report = f"ABSTAIN: {error}"
                    audit = None

            st.text(report)
            if audit:
                with st.expander("Tool trace", expanded=True):
                    render_agent_trace(audit)
                render_agent_outputs(audit)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": report,
            "audit": audit,
        })
