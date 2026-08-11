"""One local Ollama agent for the GB power risk tools."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import tools as power_tools


MODEL_NAME = "qwen3:8b"
MAX_ROUNDS = 5
MAX_TOOL_CALLS = 6
LONDON = ZoneInfo("Europe/London")
AUDIT_FILE = Path(__file__).resolve().with_name("agent_audit.json")


SYSTEM_POLICY = """
You are a GB electricity imbalance risk research agent.

Your job is to choose the shortest valid sequence of deterministic Python
tools for the user's question. You never calculate probabilities yourself.

Rules:
1. Call check_data before any date-specific prediction.
2. Call predict_risk only with the validation_id returned by check_data.
3. Call run_scenario only with a prediction_id returned by predict_risk.
4. If a tool returns ABSTAIN, do not use downstream tools for that date.
   Stop unless the user explicitly supplied another target date. In that case,
   call check_data for that stated alternative and continue only if it passes.
   Never invent or silently change a target date.
5. Use get_model_card alone for questions only about model quality.
6. Use no more tools than the question needs.
7. Never suggest a trade, order, position, or capital allocation.
8. Never invent, transform, or calculate a number. Python renders the report.
9. When the required evidence is collected, stop calling tools.
""".strip()


def _hash_payload(value) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _result_reference(result: dict) -> str | None:
    for key in ["result_id", "prediction_id", "validation_id", "model_id"]:
        if result.get(key):
            return result[key]
    return None


def _explicit_dates(question: str) -> list[str]:
    """Return unique ISO dates explicitly written by the user, in order."""
    return list(dict.fromkeys(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)))


def _abstention(reason: str, steps: list[dict]) -> str:
    tools_used = ", ".join(step["tool"] for step in steps) or "none"
    return (
        "GB POWER IMBALANCE RISK BRIEF\n\n"
        "Status: ABSTAIN\n"
        f"Reason: {reason}\n"
        f"Tools used: {tools_used}\n\n"
        "No risk conclusion was produced."
    )


def _render_report(question: str, steps: list[dict]) -> str:
    if not steps:
        return _abstention("The agent collected no deterministic evidence", steps)

    # A failed primary date can be resolved by a later successful check of an
    # alternative date explicitly supplied by the user. The full history still
    # remains in the audit, while the final result for each tool controls status.
    latest_steps = {}
    for step in steps:
        latest_steps[step["tool"]] = step

    failed = [
        step for step in latest_steps.values()
        if step["result"].get("status") == "ABSTAIN"
    ]
    if failed:
        result = failed[-1]["result"]
        reason = result.get("reason")
        if not reason:
            reason = "; ".join(result.get("critical_issues", []))
        return _abstention(reason or "A critical check failed", steps)

    results = {
        name: step["result"] for name, step in latest_steps.items()
    }
    lines = [
        "GB POWER IMBALANCE RISK BRIEF",
        "",
        "Status: READY FOR HUMAN REVIEW",
        f"Question: {question}",
        "Tools used: " + " -> ".join(step["tool"] for step in steps),
    ]

    check = results.get("check_data")
    if check:
        failed_checks = [
            step for step in steps
            if step["tool"] == "check_data"
            and step["result"].get("status") == "ABSTAIN"
        ]
        if failed_checks:
            lines.extend(["", "Fallback routing"])
            for step in failed_checks:
                failed_result = step["result"]
                failed_date = failed_result.get(
                    "target_date", step["arguments"].get("target_date", "unknown")
                )
                lines.append(
                    f"{failed_date}: ABSTAIN, "
                    f"{failed_result.get('reason', 'validation failed')}"
                )
            lines.append(
                f"Alternative date selected after validation: {check['target_date']}"
            )
        lines.extend([
            "",
            "Data check",
            f"Target date: {check['target_date']}",
            f"Observed periods: {check['observed_periods']} of {check['expected_periods']}",
            f"Cutoff: {check['cutoff_utc']}",
            f"Evidence: {check['validation_id']}",
        ])

    prediction = results.get("predict_risk")
    if prediction:
        summary = prediction["summary"]
        lines.extend([
            "",
            "Model result",
            f"Mean short-system probability: {summary['mean_short_probability_pct']}%",
            f"Maximum probability: {summary['maximum_short_probability_pct']}%",
            f"Minimum probability: {summary['minimum_short_probability_pct']}%",
            f"Evidence: {prediction['prediction_id']}",
            "",
            "Highest-risk settlement periods",
        ])
        for period in prediction["highest_risk_periods"]:
            lines.append(
                f"SP{period['settlement_period']} at {period['start_time_local']}: "
                f"{period['champion_probability_pct']}% short probability, "
                f"daily rank {period['daily_risk_rank']}"
            )

    scenario = results.get("run_scenario")
    if scenario:
        summary = scenario["summary"]
        lines.extend([
            "",
            f"Scenario: {scenario['scenario']}",
            f"Baseline mean probability: {summary['baseline_mean_probability_pct']}%",
            f"Scenario mean probability: {summary['scenario_mean_probability_pct']}%",
            f"Mean change: {summary['mean_change_pp']} percentage points",
            f"Maximum absolute period change: {summary['maximum_absolute_change_pp']} percentage points",
            f"Evidence: {scenario['result_id']}",
        ])

    card = results.get("get_model_card")
    if card:
        metrics = card.get("test_metrics", {})
        lines.extend([
            "",
            "Model card",
            f"Champion: {card['champion']}",
            f"Training rows: {card['training_rows']}",
            f"Evaluation period: {card['evaluation_period']}",
        ])
        if "auc" in metrics:
            lines.append(f"Test AUC: {metrics['auc']}")
        if "brier" in metrics:
            lines.append(f"Test Brier score: {metrics['brier']}")
        lines.append(f"Evidence: {card['result_id']}")

    lines.extend([
        "",
        "This is decision support, not a trading instruction.",
    ])
    return "\n".join(lines)


def _save_audit(audit: dict) -> None:
    AUDIT_FILE.write_text(
        json.dumps(audit, indent=2, default=str), encoding="utf-8"
    )


def run_agent(question: str, chat_function=None) -> tuple[str, dict]:
    """Run the bounded tool-calling loop and return the grounded report."""
    if chat_function is None:
        try:
            from ollama import chat as chat_function
        except ImportError as error:
            raise RuntimeError(
                "The ollama Python package is missing. Run: pip install ollama"
            ) from error

    now = datetime.now(LONDON)
    tomorrow = (now.date() + timedelta(days=1)).isoformat()
    dated_context = (
        f"Current London time: {now.isoformat(timespec='minutes')}. "
        f"Tomorrow's settlement date: {tomorrow}."
    )
    messages = [
        {"role": "system", "content": SYSTEM_POLICY + "\n\n" + dated_context},
        {"role": "user", "content": question},
    ]
    steps = []
    seen_calls = set()
    stop_reason = None
    explicit_dates = _explicit_dates(question)

    for _ in range(MAX_ROUNDS):
        try:
            response = chat_function(
                model=MODEL_NAME,
                messages=messages,
                tools=list(power_tools.TOOLS.values()),
                think=False,
                options={"temperature": 0},
            )
        except Exception as error:
            stop_reason = f"Ollama request failed: {error}"
            break

        messages.append(response.message)
        calls = response.message.tool_calls or []
        if not calls:
            stop_reason = "evidence_complete"
            break

        for call in calls:
            if len(steps) >= MAX_TOOL_CALLS:
                stop_reason = "tool_call_limit"
                break

            name = call.function.name
            arguments = dict(call.function.arguments or {})
            call_key = (name, json.dumps(arguments, sort_keys=True, default=str))
            if call_key in seen_calls:
                stop_reason = f"Repeated tool call blocked: {name}"
                break
            seen_calls.add(call_key)

            function = power_tools.TOOLS.get(name)
            if function is None:
                stop_reason = f"Unknown tool blocked: {name}"
                break

            started = perf_counter()
            try:
                inspect.signature(function).bind(**arguments)
                result = function(**arguments)
            except Exception as error:
                result = {
                    "tool": name,
                    "status": "ABSTAIN",
                    "reason": f"Tool call rejected: {error}",
                }
            elapsed_ms = round((perf_counter() - started) * 1000, 1)
            step = {
                "sequence": len(steps) + 1,
                "tool": name,
                "arguments": arguments,
                "result_id": _result_reference(result),
                "status": result.get("status"),
                "result_hash": _hash_payload(result),
                "elapsed_ms": elapsed_ms,
                "result": result,
            }
            steps.append(step)
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, default=str),
            })

            if result.get("status") == "ABSTAIN":
                attempted_dates = {
                    str(item["arguments"].get("target_date"))
                    for item in steps
                    if item["tool"] == "check_data"
                }
                untried_dates = [
                    value for value in explicit_dates
                    if value not in attempted_dates
                ]
                if name == "check_data" and untried_dates:
                    messages.append({
                        "role": "system",
                        "content": (
                            "The attempted date failed validation. Do not use "
                            "downstream tools for it. The user explicitly supplied "
                            f"this untried alternative date: {untried_dates[0]}. "
                            "Call check_data for that date next."
                        ),
                    })
                    break
                stop_reason = "critical_tool_failure"
                break

        if stop_reason not in [None, "evidence_complete"]:
            break

    if stop_reason is None:
        stop_reason = "round_limit"

    if stop_reason not in ["evidence_complete", "critical_tool_failure"]:
        report = _abstention(stop_reason, steps)
    else:
        report = _render_report(question, steps)
    status = (
        "ABSTAIN" if "Status: ABSTAIN" in report
        else "READY_FOR_HUMAN_REVIEW"
    )
    audit = {
        "run_id": "run_" + _hash_payload({
            "question": question,
            "started": now.isoformat(),
        }),
        "started_at": now.isoformat(),
        "ollama_model": MODEL_NAME,
        "system_prompt_hash": _hash_payload(SYSTEM_POLICY),
        "question": question,
        "steps": steps,
        "final_status": status,
        "stop_reason": stop_reason,
    }
    _save_audit(audit)
    return report, audit


def main() -> None:
    default_question = "Assess tomorrow's GB power imbalance risk."
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input(
            f"Research question [{default_question}]: "
        ).strip() or default_question

    try:
        report, _ = run_agent(question)
    except RuntimeError as error:
        print(f"ABSTAIN: {error}")
        return

    print("\n" + report)
    print(f"\nAudit saved to {AUDIT_FILE.name}")


if __name__ == "__main__":
    main()