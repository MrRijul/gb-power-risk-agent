"""One bounded agent for the GB power risk tools."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

import tools as power_tools


OLLAMA_MODEL = "qwen3:8b"
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_ROUNDS = 7
MAX_TOOL_CALLS = 7
LONDON = ZoneInfo("Europe/London")
AUDIT_FILE = Path(__file__).resolve().with_name("agent_audit.json")


SYSTEM_POLICY = """
You are a GB electricity imbalance risk research agent.

Your job is to choose the shortest valid sequence of deterministic Python
tools for the user's question. You never calculate probabilities yourself.

Rules:
1. Call check_data before any date-specific prediction.
2. Call predict_risk only with the validation_id returned by check_data.
3. Call compare_with_actuals only with a prediction_id returned by predict_risk,
   only when the user asks for realised outcomes, and only for a completed date.
4. Call run_scenario only with a prediction_id returned by predict_risk.
   For a natural-language sensitivity, use scenario="custom" and pass the
   user's stated demand_change_pct, wind_change_pct, and margin_change_mw.
   An omitted change is zero. Preserve signs exactly. Python enforces bounds.
5. If a tool returns ABSTAIN, do not use downstream tools for that date.
   Stop unless the user explicitly supplied another target date. In that case,
   call check_data for that stated alternative and continue only if it passes.
   Never invent or silently change a target date.
6. Use get_model_card alone for questions only about model quality.
7. Use no more tools than the question needs.
8. Never suggest a trade, order, position, or capital allocation.
9. Never invent, transform, or calculate a number. Python renders the report.
10. When the required evidence is collected, stop calling tools.
""".strip()


def _hash_payload(value) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _result_reference(result: dict) -> str | None:
    for key in ["result_id", "prediction_id", "validation_id", "model_id"]:
        if result.get(key):
            return result[key]
    return None


def _agent_view(result: dict) -> dict:
    """Keep full evidence in the audit but send compact results to the LLM."""
    compact = dict(result)
    compact.pop("period_predictions", None)
    compact.pop("period_results", None)
    return compact


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

    comparison = results.get("compare_with_actuals")
    if comparison:
        summary = comparison["summary"]
        lines.extend([
            "",
            "Forecast versus actual outcome",
            f"Mean predicted probability: {summary['mean_predicted_probability_pct']}%",
            f"Actual short-period share: {summary['actual_short_share_pct']}%",
            f"Accuracy at 50% threshold: {summary['accuracy_at_50pct_pct']}%",
            f"Daily Brier score: {summary['daily_brier_score']}",
            f"Evidence: {comparison['result_id']}",
        ])

    scenario = results.get("run_scenario")
    if scenario:
        summary = scenario["summary"]
        assumptions = scenario.get("assumptions", {})
        if scenario["scenario"] == "custom":
            assumption_text = (
                f"demand {assumptions.get('demand_change_pct', 0):+g}%, "
                f"wind {assumptions.get('wind_change_pct', 0):+g}%, "
                f"margin {assumptions.get('margin_change_mw', 0):+g} MW"
            )
        else:
            assumption_text = ", ".join(
                f"{name} multiplier {value:g}"
                for name, value in assumptions.items()
            )
        lines.extend([
            "",
            "Model sensitivity scenario",
            f"Scenario assumptions: {assumption_text}",
            f"Baseline mean probability: {summary['baseline_mean_probability_pct']}%",
            f"Scenario mean probability: {summary['scenario_mean_probability_pct']}%",
            f"Mean change: {summary['mean_change_pp']} percentage points",
            f"Maximum absolute period change: {summary['maximum_absolute_change_pp']} percentage points",
            f"Evidence: {scenario['result_id']}",
            "",
            "Most affected settlement periods",
        ])
        for period in scenario["most_changed_periods"]:
            lines.append(
                f"SP{period['settlement_period']}: "
                f"{period['baseline_probability_pct']}% to "
                f"{period['scenario_probability_pct']}% "
                f"({period['change_pp']:+g} pp)"
            )
        lines.append(scenario["method_note"])

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


PARAMETER_SCHEMAS = {
    "target_date": {
        "type": "string",
        "description": "Settlement date in YYYY-MM-DD format.",
    },
    "validation_id": {
        "type": "string",
        "description": "Identifier returned by check_data for the same date.",
    },
    "prediction_id": {
        "type": "string",
        "description": "Identifier returned by predict_risk.",
    },
    "scenario": {
        "type": "string",
        "enum": ["custom"],
        "description": "Use custom for a bounded model sensitivity.",
    },
    "demand_change_pct": {
        "type": "number",
        "minimum": -10,
        "maximum": 10,
        "description": "Demand change in percent, from -10 to +10.",
    },
    "wind_change_pct": {
        "type": "number",
        "minimum": -30,
        "maximum": 30,
        "description": "Wind-generation change in percent, from -30 to +30.",
    },
    "margin_change_mw": {
        "type": "number",
        "minimum": -2000,
        "maximum": 2000,
        "description": "Margin change in MW, from -2000 to +2000.",
    },
}


def _groq_tools() -> list[dict]:
    """Convert the existing Python tool registry into Groq JSON schemas."""
    schemas = []
    for name, function in power_tools.TOOLS.items():
        signature = inspect.signature(function)
        properties = {}
        required = []
        for parameter_name, parameter in signature.parameters.items():
            properties[parameter_name] = dict(
                PARAMETER_SCHEMAS[parameter_name]
            )
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)
            else:
                properties[parameter_name]["default"] = parameter.default

        description = (inspect.getdoc(function) or name).splitlines()[0]
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        })
    return schemas


def _select_backend(chat_function=None):
    """Use Groq when configured, otherwise retain the local Ollama path."""
    if chat_function is not None:
        return "ollama", OLLAMA_MODEL, chat_function

    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
        except ImportError as error:
            raise RuntimeError(
                "The groq package is missing. Run: pip install groq"
            ) from error
        return "groq", GROQ_MODEL, Groq()

    try:
        from ollama import chat
    except ImportError as error:
        raise RuntimeError(
            "No GROQ_API_KEY was found and the ollama package is missing."
        ) from error
    return "ollama", OLLAMA_MODEL, chat


def run_agent(question: str, chat_function=None) -> tuple[str, dict]:
    """Run the bounded tool-calling loop and return the grounded report."""
    provider, model_name, model_client = _select_backend(chat_function)

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
            if provider == "groq":
                response = model_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    tools=_groq_tools(),
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    temperature=0,
                )
                response_message = response.choices[0].message
            else:
                response = model_client(
                    model=model_name,
                    messages=messages,
                    tools=list(power_tools.TOOLS.values()),
                    think=False,
                    options={"temperature": 0},
                )
                response_message = response.message
        except Exception as error:
            stop_reason = f"{provider.title()} request failed: {error}"
            break

        messages.append(response_message)
        calls = response_message.tool_calls or []
        if not calls:
            stop_reason = "evidence_complete"
            break

        for call in calls:
            if len(steps) >= MAX_TOOL_CALLS:
                stop_reason = "tool_call_limit"
                break

            name = call.function.name
            try:
                if provider == "groq":
                    arguments = json.loads(call.function.arguments or "{}")
                    if arguments is None:
                        arguments = {}
                else:
                    arguments = dict(call.function.arguments or {})
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must be a JSON object")
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                stop_reason = f"Invalid tool arguments blocked: {name}: {error}"
                break
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
            tool_message = {
                "role": "tool",
                "content": json.dumps(_agent_view(result), default=str),
            }
            if provider == "groq":
                tool_message["tool_call_id"] = call.id
                tool_message["name"] = name
            else:
                tool_message["tool_name"] = name
            messages.append(tool_message)

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
        "llm_provider": provider,
        "llm_model": model_name,
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
