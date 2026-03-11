# Genetic Algorithm for Spinopelvic Alignment Surgery Planning

## Environment Setup

Requires [Poetry](https://python-poetry.org/docs/#installation) (v2.0+) and **Python 3.11+**.

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
cd Data-Science-Capstone
poetry install

# Activate the virtual environment
source .venv/bin/activate

# Verify
which python   # should point to .venv/bin/python
```

To add a new package:

```bash
poetry add <package-name>
```

---

## Project Structure

```
Data-Science-Capstone/
├── pyproject.toml              # Poetry setup
├── src/                        # Module files
│   ├── config.py               # Data paths, decision variables, model paths
│   ├── optimization_utils.py   # GA utility functions
│   ├── problem.py              # pymoo problem definition
│   ├── runs.py                 # Scenario presets & run_optimization()
│   ├── scoring.py              # Composite scoring, GAP score, alignment
│   ├── solutions.py            # Diversity selection for top-N plans
│   └── display.py              # All display functions
├── notebooks/                  # Jupyter notebooks
│   ├── 00_data_cleaning.ipynb
│   ├── 01_mech_failure_model.ipynb
│   ├── 02_delta_models.ipynb
│   └── 03_optimization.ipynb
├── data/
│   ├── raw/                    # Raw data files here
│   ├── processed/              # Cleaned train + holdout CSVs
│   └── intermediate/           # Intermediate data
├── artifacts/                  # Saved model .joblib files 
└── README.md
```

---

## Data Setup

### Add a new raw data file

Place your source Excel file in `data/raw/`. Then update the path in **`src/config.py`**:

```python
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "FILE_NAME.xlsx"
```

The data-cleaning notebook (`00_data_cleaning.ipynb`) reads from this path. If data values have changed, may need to do some additional data cleaning.

### Specifying holdout patient IDs

Holdout patients are separated from the training set in `00_data_cleaning.ipynb`. To change which patients are held out, edit `HOLDOUT_IDS` and `PATIENT_DESCRIPTIONS` in **`src/config.py`**:

```python
HOLDOUT_IDS = [1176294, 2964021, 818588, 6380632]
```

These patients are saved to `data/processed/holdout_patients.csv` and excluded from model training. They are later used in `03_optimization.ipynb` to compare the optimizer's recommendations against the actual surgical outcomes.

---

## Notebook Pipeline

Run the notebooks **in order**. Each notebook depends on outputs from the previous one.

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `00_data_cleaning.ipynb` | Loads raw Excel data, cleans columns, splits into train/holdout sets, saves to `data/processed/`. |
| 2 | `01_mech_failure_model.ipynb` | Trains and compares mechanical failure classification models. Saves best model artifact to `artifacts/MechanicalFailure/`. |
| 3 | `02_delta_models.ipynb` | Trains regression models for each delta alignment parameter (LL, SS, L4S1, GlobalTilt, T4PA, L1PA, SVA) and ODI. Saves model artifacts to `artifacts/`. |
| 4 | `03_optimization.ipynb` | Loads all trained models, runs the GA optimization for multiple scenarios, and displays results. |

---

## Running the Optimization Notebook (`03_optimization.ipynb`)

### 1. GA Configuration

The configuration cell at the top of the notebook controls all tunable parameters:

```python
POP_SIZE  = 100    # GA population size
N_GEN     = 20     # Number of generations
SEED      = 42     # Random seed for reproducibility
TOP_N     = 4      # Number of diverse solutions to display
SCORE_TOL = 5      # Score window around best for diversity selection

PSO_LL_OVERRIDE = False  # Clamp delta_LL to literature range when PSO is present
DEDUP_BEST = True          # Deduplicate best plans across scenarios
```

### 2. Selecting Scenarios (Cell 10)

Choose which optimization scenarios to run by editing the `SCENARIOS` list. Presets are defined in `src/runs.py`:

- `"equal"` — All alignment components weighted equally
- `"mech_fail"` — Primarily mechanical failure risk
- `"mech_fail_t4l1pa"` — Mech failure + T4PA-L1PA mismatch
- `"l4s1"` — Primarily L4-S1 in ideal range
- `"t4l1pa"` — Primarily T4PA-L1PA mismatch
- `"gap_score"` — Primarily GAP score improvement
- `"gap_score_mech_fail"` — GAP score + mechanical failure
- `"ll"` — Primarily PI-LL mismatch
- `"odi"` — Primarily lowest predicted postop ODI

To add a new scenario, add a new entry in `src/runs.py`, similar to these.

### 3. Selecting Patients

Each patient section begins with a cell that sets the patient ID and loads their data from the holdout file:

```python
PATIENT_ID = 6380632
patient_fixed = ou.load_patient_data(patient_id=PATIENT_ID, data_path=config.DATA_HOLDOUT)
```

To add more patients, duplicate one of the existing patient sections (selection → run optimization → display results) and change the patient ID and variable suffixes.

### 4. `DEDUP_BEST` — Deduplicate Best Plans

When `DEDUP_BEST = True`, the "Best Solution per Scenario" table ensures each scenario displays a distinct surgical plan. If two scenarios both select the same best plan, the second one will fall back to its second best diverse alternative, so you can see a wider range of recommendations at a glance.

Set `DEDUP_BEST = False` to allow the same plan to appear as "best" in multiple scenarios.

### 5. `PSO_LL_OVERRIDE` — Delta LL Correction

When `PSO_LL_OVERRIDE = True`, if the surgical plan includes a PSO/osteotomy, the predicted delta LL is clamped to the clinically expected range of **20–45°**.

---

## Key Outputs

- **Best Solution per Scenario** — One table showing each scenario's single best plan alongside the actual surgical plan (using real postop values from holdout data).
- **4 Best Solutions per Scenario** — Per-scenario tables with the actual plan (highlighted) and top diverse optimizer recommendations.