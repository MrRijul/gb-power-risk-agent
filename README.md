# GB Power Imbalance Risk Agent
I built this project to explore a practical question: using only information available at 16:00 on the previous day, can public Elexon forecasts help identify which GB settlement periods are more likely to be short?

## Project Highlights

The model combines demand, wind generation, system margin, indicated imbalance and time-of-day information. I chose an Extremely Randomized Trees classifier because these variables are likely to interact in nonlinear ways. I also required at least 100 observations in every terminal leaf to reduce the risk of fitting narrow, unstable half-hour patterns.

I trained the candidate models on 2022–2023, used 2024 to select between Extra Trees and penalised logistic regression, and kept 2025 untouched for the final out-of-sample test.

On the 2025 test set, Extra Trees achieved an AUC of 0.632 and improved the Brier score by 5.7% relative to a constant probability forecast. This is a useful but moderate forecasting result.

The main purpose of the project is to show how an agentic layer turns the model into a repeatable research workflow. It checks whether the data were genuinely available at the forecast cutoff, runs the model and bounded scenarios, compares historical forecasts with realised outcomes, and records every tool call in an audit trace. The language model coordinates the workflow, while deterministic Python produces every calculation.

### ML Note

I use an an Extremely Randomized Trees classifier to model a likely non-linear and interaction dependent relationship between demand, wind generation, margin, indicated imbalance and time of day. I used a conservative minimum leaf size of 100 to avoid fitting narrow half-hour patterns. The model was selected against penalised logistic regression using 2024 Brier score and then evaluated on an untouched 2025 test set. 

I train models on 2022-2023, 2024 to select the model, and 2025 was an out of sample test, to see if it holds up.

The final result achieves an AUC of 0.632 and a 5.7% Brier-score improvement over a constant forecast. The purpose of this is to show agentic capability.

## Method and File Guide

The first two notebooks build a point-in-time history from Elexon. The third notebook trains a transparent benchmark and a nonlinear probability model, then evaluates them on a later year.

### Current files

```text
01_build_project.ipynb   First data pull and leakage audit
02_build_history.ipynb   Clock-change test and historical downloader
03_train_model.ipynb     Chronological model training and evaluation
tools.py                 Data checks, prediction, scenarios and model card
agent.py                 One local Ollama tool-calling agent
app.py                   Streamlit dashboard and local agent chat
requirements.txt         Packages needed to run the notebook
README.md                This explanation
```

The notebooks create:

- `seven_day_sample.csv` after Step 1 passes
- `dst_audit_sample.csv` after the Step 2 test passes
- `power_history_raw.csv` with the full expected historical schedule
- `excluded_dates.csv` documenting dates with incomplete source data
- `power_history.csv` containing the historical schedule
- `power_risk_model.joblib` containing the fitted model and benchmark
- `model_metrics.csv` containing validation and test results
- `test_predictions.csv` containing the untouched 2025 predictions
- `feature_importance.csv` containing the model explanation

The included test run produced:

- 336 settlement periods
- 0 duplicate keys
- 0 missing critical values
- 0 forecasts published after the cutoff
- 59.52% of settlement periods classified as system short

### Setup on Windows

Open a terminal inside this folder and run:

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Open the folder in VS Code. Select **Python (GB Power Risk)** as the notebook kernel. Open `01_build_project.ipynb` and run the cells from top to bottom. Once it prints PASS, do the same with `02_build_history.ipynb`.

Leave `RUN_FULL_HISTORY = False` for the first Step 2 run. It downloads 14 dates around the two 2025 clock changes. The expected result is:

```text
PASS: settlement dates, clock changes and publication cutoffs are correct.
```

That test produces 672 rows: one 46-period day, one 50-period day and twelve normal 48-period days.

After it passes, change this line in the Step 2 settings cell:

```python
RUN_FULL_HISTORY = True
```

Run the notebook again to create the 2022 to 2025 model history. Completed monthly chunks are stored in `cache`, so the download can resume after an interruption.

The audited full run found four dates with incomplete forecast features and ten isolated rows with missing outcomes. Nothing is filled. Step 3 applies the exclusion log again, removes the four feature-outage dates in full, and removes the ten rows without a target. This leaves 69,926 modelling observations. The live agent will abstain whenever a required forecast is unavailable.

### Step 3: train the model

Keep these three files in the same folder:

```text
03_train_model.ipynb
power_history.csv
excluded_dates.csv
```

Run this once after downloading the updated requirements:

```powershell
pip install -r requirements.txt
```

Then open `03_train_model.ipynb`, select the same project kernel, and use **Run All**.

The time split is fixed:

- 2022 to 2023 for training
- 2024 for choosing between the models
- 2025 as the untouched final test

The included run selected Extra Trees over logistic regression. On 2025 it produced:

- AUC: 0.632
- Brier score: 0.235, compared with 0.250 for a constant forecast
- Brier improvement over the constant forecast: 5.7%
- Observed short-system rate: 24.4% in the lowest-risk decile and 65.8% in the highest-risk decile

This is a moderate risk signal. It is not presented as a trading strategy or proof of profit.

## What the notebook checks

- 48 settlement periods per normal day
- No duplicate date and settlement-period keys
- No missing demand, wind, indicated imbalance, margin or outcome values
- Every forecast publication timestamp is before the 16:00 London cutoff on the previous day
- The target is created only from realised Net Imbalance Volume
- Clock-change days contain the correct 46 or 50 settlement periods
- Every source agrees with Elexon's settlement-date and settlement-period key

The target is:

```python
system_short = net_imbalance_volume > 0
```

Positive Net Imbalance Volume means the GB system was short.

#### Data source

The data come from the public Elexon Insights API. No API key is required.

- Developer portal: https://developer.data.elexon.co.uk/
- System price explanation: https://bmrs.elexon.co.uk/system-prices

### Step 4: run the local agent

Install the Ollama desktop application, then download the local model once:

```powershell
ollama pull qwen3:8b
```

Make sure the model, data and Python files are in the same folder:

```text
power_risk_model.joblib
power_history.csv
excluded_dates.csv
model_metrics.csv
tools.py
agent.py
```

Run the agent from the VS Code terminal:

```powershell
python agent.py
```

You can also give it a question directly:

```powershell
python agent.py "Assess tomorrow. What if demand rises 6% and wind falls 12%?"
```

The local LLM chooses which tools are needed. Python enforces the order, performs every calculation, stops on missing or late data, and renders the final numbers from structured results. Custom scenarios are bounded to demand changes of plus or minus 10%, wind changes of plus or minus 30%, and margin changes of plus or minus 2,000 MW. The agent saves its complete tool sequence in `agent_audit.json`.

For a completed historical date, the agent can also compare its point-in-time probabilities with realised Net Imbalance Volume. This is an ex-post review. The model predicts the probability of a short system, not the magnitude of NIV.

## Step 5: run the dashboard

Start the dashboard from the project folder:

```powershell
python -m streamlit run app.py
```

The Risk dashboard tab provides manual data checks, probability charts, realised-outcome comparison and bounded scenario sliders. The Research agent tab sends natural-language questions to the local Ollama agent and shows the tool trace. The chat tab requires Ollama on the same computer. The manual dashboard does not.
