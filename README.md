# GB Power Imbalance Risk Agent

I built this project around a practical question: using only information available at 16:00 London time on the previous day, can public Elexon forecasts help identify which GB settlement periods are more likely to be short?

**Live application:** [gb-power-risk-agent.streamlit.app](https://gb-power-risk-agent.streamlit.app/)

The model estimates the probability of a short electricity system for each half-hour of the following settlement day. It does not predict the size of Net Imbalance Volume, recommend trades or update itself automatically.

## What the Project Does

For a selected settlement date, the workflow:

1. Fetches the demand, wind, system-margin and indicated-imbalance forecasts available by the D-1 16:00 London cutoff.
2. Checks publication times, missing values, stale forecasts, duplicate keys and the expected 46, 48 or 50 settlement periods.
3. Stops if a critical data check fails.
4. Uses a frozen Extra Trees model to estimate short-system probability for every settlement period.
5. Optionally runs bounded demand, wind and margin sensitivity scenarios.
6. For a completed date, fetches realised Net Imbalance Volume and compares the forecast with the outcome.
7. Records the tool sequence, inputs, outputs and stop reason in an audit trace.

The language model decides which tools are relevant to the research question. All data checks, features, probabilities, comparisons and scenario calculations come from deterministic Python.

## Core Functions

| Function | What it does |
| --- | --- |
| `check_data(target_date)` | Fetches the eligible Elexon forecasts for the selected date and confirms that they were available by the cutoff. It checks completeness, freshness, publication timing, settlement-period structure and source alignment. It returns `ABSTAIN` if the date is too early or any critical input fails. |
| `predict_risk(target_date, validation_id)` | Uses the validated data snapshot and frozen Extra Trees model to estimate a probability for each settlement period. It also returns the daily mean, minimum, maximum and six highest-risk periods. It does not retrain the model. |
| `run_scenario(prediction_id, ...)` | Changes demand, wind and margin within fixed bounds, reruns the model and compares the scenario probabilities with the baseline. Other inputs remain fixed, so this is a model sensitivity rather than a new market forecast or causal estimate. |
| `compare_with_actuals(prediction_id)` | For a completed day, fetches realised Elexon Net Imbalance Volume and compares it with the saved point-in-time probabilities. It reports period-level outcomes, a daily Brier score and accuracy at a 50% classification threshold. |
| `get_model_card()` | Returns the model design, development periods, features, test results and limitations without running a date-specific forecast. |
| `run_agent(question)` | Allows one bounded language model to choose the shortest valid tool sequence. Python enforces tool order, scenario limits, tool-call limits and abstention before saving the full trace to `agent_audit.json`. |

## Evidence and Tool Order

The tools are linked through evidence identifiers.

A prediction requires the `validation_id` produced by a successful data check. Scenario analysis and realised-outcome review then require the `prediction_id` produced by the forecast.

```text
check_data()
      |
      v
validation_id
      |
      v
predict_risk()
      |
      v
prediction_id
      |
      +----------------------+
      |                      |
      v                      v
run_scenario()       compare_with_actuals()
```

This structure prevents the agent from skipping required validation or running a scenario against an unidentified forecast.

## Example Research Questions

```text
Assess tomorrow's GB short-system risk.

Assess 2026-08-13. What if demand is 6% higher and wind generation is 12% lower?

Compare the forecast for 2025-11-20 with the realised outcome.

What are the model's test results and limitations?

Is the Elexon data for tomorrow complete and safe to use?
```

The agent only calls the tools needed to answer the question. For example, a question about model limitations should call the model card without downloading a live forecast.

## Model and Results

The model combines 15 features built from:

- Forecast electricity demand
- Forecast wind generation
- Forecast system margin
- Forecast indicated imbalance
- Residual demand
- Proportional demand, wind, margin and imbalance measures
- Time-of-day, weekday and seasonal information

I selected an **Extremely Randomized Trees classifier**, also known as an **Extra Trees classifier**, because the relationships between these variables are likely nonlinear and interaction-dependent.

The model contains 300 randomized decision trees. Each terminal leaf must contain at least 100 observations, which reduces the risk of fitting narrow and unstable half-hour patterns. Penalised logistic regression is used as a transparent benchmark.

### Chronological Evaluation

The evaluation follows the order in which the information would have become available:

- **2022 to 2023:** model training
- **2024:** model selection
- **2025:** untouched final test
- **After evaluation:** the selected model was refitted through 2025 and frozen for live use

After documented source exclusions, the modelling dataset contains **69,926 settlement-period observations**.

On the untouched 2025 test set, Extra Trees produced:

| Metric | Result |
| --- | ---: |
| AUC | 0.632 |
| Brier score | 0.235 |
| Constant-forecast Brier score | 0.250 |
| Brier-score improvement | 5.7% |
| Short rate in lowest-risk decile | 24.4% |
| Short rate in highest-risk decile | 65.8% |

The result represents a moderate out-of-sample risk signal. It is not presented as a trading strategy or proof of profitability.

The main contribution of the project is showing how a forecasting model can be turned into a repeatable and auditable research workflow.

## Point-in-Time Safeguards

The historical and live pipelines check that:

- Forecasts were published by 16:00 London time on D-1.
- Normal days contain 48 settlement periods.
- Spring and autumn clock-change days contain 46 and 50 periods respectively.
- Date and settlement-period keys are unique.
- Required forecasts and publication timestamps are present.
- Forecasts are not stale or published after the cutoff.
- Source start times agree with Elexon's settlement keys.
- Missing inputs and outcomes are excluded rather than imputed.

The target is defined only from the realised outcome:

```python
system_short = net_imbalance_volume > 0
```

Positive Net Imbalance Volume means that the GB electricity system was short.

## Scenario Analysis

The scenario tool allows the user or agent to change selected inputs within fixed Python-enforced limits:

| Input | Permitted range |
| --- | ---: |
| Demand | -10% to +10% |
| Wind generation | -30% to +30% |
| System margin | -2,000 MW to +2,000 MW |

For example:

```text
What happens if demand is 7% higher, wind generation is 15% lower and margin falls by 800 MW?
```

The tool reruns the model with those adjusted inputs and reports:

- Baseline mean probability
- Scenario mean probability
- Change in percentage points
- Largest individual-period change
- Most affected settlement periods
- Full baseline and scenario probability profiles

These outputs are model sensitivities. They hold the remaining inputs constant and should not be interpreted as causal estimates or new market forecasts.

## Streamlit Interface

The Streamlit application provides two ways to use the project.

### Research Agent

The research agent accepts natural-language questions and can:

- Validate the data for a selected date
- Generate a short-system risk assessment
- Identify the highest-risk settlement periods
- Run bounded custom scenarios
- Compare historical forecasts with realised outcomes
- Explain model performance and limitations
- Display the selected tools and audit trace
- Abstain when the evidence is incomplete

### Risk Dashboard

The manual dashboard allows users to:

- Select a settlement date
- Run the point-in-time data checks
- Generate settlement-period probabilities
- View the probability profile using London time
- Compare completed dates with actual outcomes
- Adjust demand, wind and margin using scenario sliders
- Compare the baseline and scenario probability curves

The public deployment uses a Groq-hosted language model for tool routing. The language model does not calculate or modify model outputs. A local Ollama model remains available as an optional fallback.

## Project Files

```text
01_build_project.ipynb   Small data pull and first leakage audit
02_build_history.ipynb   DST-safe historical downloader and exclusion log
03_train_model.ipynb     Chronological training, selection and evaluation
tools.py                 Deterministic validation, prediction and review tools
agent.py                 Bounded LLM routing, reporting and audit trace
app.py                   Streamlit research agent and risk dashboard
power_risk_model.joblib  Frozen fitted model and model metadata
model_metrics.csv        Validation and test results
feature_importance.csv   Model feature importance
excluded_dates.csv       Documented source-data exclusions
requirements.txt         Python dependencies
README.md                Project documentation
```

## Data Source

The data come from the public Elexon Insights API. No Elexon API key is required.

- [Elexon Insights developer portal](https://developer.data.elexon.co.uk/)
- [Elexon system-price and NIV explanation](https://bmrs.elexon.co.uk/system-prices)

## Limitations

- The model predicts the probability of a short system, not the magnitude of Net Imbalance Volume.
- It does not predict electricity prices or produce trading recommendations.
- The 2025 result is moderate and does not establish profitability.
- Settlement periods within the same day are related, so 69,926 rows are not 69,926 independent events.
- Live dates after 2025 are genuine post-test use, but performance should be monitored for model drift.
- Scenario outputs are bounded model sensitivities, not causal forecasts.
- The production model is frozen and does not retrain itself when fresh forecasts are fetched.
- The Extra Trees model can capture nonlinear patterns but is less directly interpretable than logistic regression.
- Forecast quality depends on the completeness and reliability of the underlying Elexon publications.

<details>
<summary><strong>Optional: Run the Application Locally</strong></summary>

### Create the Environment

Open PowerShell inside the project folder:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Use Groq Locally

Set the Groq API key for the current PowerShell session:

```powershell
$env:GROQ_API_KEY="your-key"
python -m streamlit run app.py
```

Do not save the API key inside the code or commit it to GitHub.

### Use Ollama Locally

Install Ollama and download the local model once:

```powershell
ollama pull qwen3:8b
python -m streamlit run app.py
```

If `GROQ_API_KEY` is absent, the agent uses Ollama as its local fallback.

The manual risk dashboard can still run without a language model because all calculations are contained in the deterministic Python tools.

### Run the Agent Without Streamlit

```powershell
python agent.py
```

You can also provide the question directly:

```powershell
python agent.py "Assess tomorrow. What if demand rises 6% and wind falls 12%?"
```

</details>

<details>
<summary><strong>Optional: Rebuild the Data and Model</strong></summary>

Run the notebooks in order.

### 1. Initial Data and Leakage Test

Open:

```text
01_build_project.ipynb
```

This notebook downloads a small Elexon sample and checks the basic point-in-time logic.

### 2. Historical Data Construction

Open:

```text
02_build_history.ipynb
```

The first run tests dates around the spring and autumn clock changes.

Keep:

```python
RUN_FULL_HISTORY = False
```

The expected result is:

```text
PASS: settlement dates, clock changes and publication cutoffs are correct.
```

After the test passes, change the setting to:

```python
RUN_FULL_HISTORY = True
```

Run the notebook again to build the 2022 to 2025 history. Completed monthly downloads are stored in `cache`, allowing the process to resume after an interruption.

The full audit found four dates with incomplete forecast features and ten isolated settlement periods without realised outcomes. Nothing was filled.

The four forecast-outage dates were removed in full, while the ten observations without a target were excluded individually. This left 69,926 modelling observations.

### 3. Model Training and Evaluation

Open:

```text
03_train_model.ipynb
```

The notebook:

1. Applies the documented exclusion log.
2. Creates the 15 model features.
3. Trains penalised logistic regression and Extra Trees.
4. Uses 2024 to select between the models.
5. Evaluates the selected model once on the untouched 2025 test set.
6. Refits the selected specification through 2025 for live use.
7. Saves the fitted model and supporting evidence.

### Generated Files

```text
power_history_raw.csv    Expected historical schedule before exclusions
excluded_dates.csv       Dates and reasons for source-data exclusions
power_history.csv        Clean point-in-time model history
power_risk_model.joblib  Final fitted model and metadata
model_metrics.csv        Model-selection and test metrics
test_predictions.csv     Untouched 2025 predictions
feature_importance.csv   Extra Trees feature importance
```

</details>
