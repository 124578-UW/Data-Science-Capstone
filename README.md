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
│   └── display.py              # Result display helpers (preop profile + best-per-scenario)
├── notebooks/                  # Jupyter notebooks
│   ├── 00_data_cleaning.ipynb
│   ├── 01_mech_failure_model.ipynb
│   ├── 02_delta_models.ipynb
│   ├── 03_optimization_holdout_patients.ipynb
│   └── 04_optimization_new_patient.ipynb
├── data/
│   ├── raw/                    # Raw data files here
│   ├── processed/              # Cleaned train + holdout CSVs
│   └── intermediate/           # Intermediate data
├── artifacts/                  # Saved model .joblib files 
└── README.md
```

---

## Data Setup

### Add a new raw data file for model training

Place your source Excel file in `data/raw/`. Then update the path in **`src/config.py`**:

```python
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "FILE_NAME.xlsx"
```

The data-cleaning notebook (`00_data_cleaning.ipynb`) reads from this path. If data values have changed, may need to do some additional data cleaning.

### Holdout Patient Selection

Holdout patients are separated from the training set in `00_data_cleaning.ipynb` using **PCA + farthest-point sampling** to ensure maximum diversity across preoperative/patient features.

**How it works:**

1. Preoperative/patient features (`PATIENT_FIXED_COLS`) are standardized (impute missing, scale numerics, encode categoricals).
2. PCA reduces the feature space to the number of components that explain ≥90% of variance.
3. A greedy farthest-point algorithm starts with patient `1176294` (always included) and iteratively selects the patient whose minimum distance to all already-selected patients is largest.
4. This repeats until `N_HOLDOUT` patients are selected (default: 10).

To configure, edit **`src/config.py`**:

```python
N_HOLDOUT = 10                            # Total holdout patients
HOLDOUT_SEED_ID = 1176294                 # Fixed seed patient always included
```

`src/config.py` stores only the selection settings (seed + holdout count). The final selected holdout ID list is generated in `00_data_cleaning.ipynb` at runtime and written to `data/processed/holdout_patients.csv`.

These patients are saved to `data/processed/holdout_patients.csv` and excluded from model training. They are later used in `03_optimization_holdout_patients.ipynb` to compare the optimizer's recommendations against the actual surgical outcomes.

---

## Notebook Pipeline

Run the notebooks **in order**. Each notebook depends on outputs from the previous one.

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `00_data_cleaning.ipynb` | Loads raw Excel data, cleans columns, splits into train/holdout sets, saves to `data/processed/`. |
| 2 | `01_mech_failure_model.ipynb` | Trains and compares mechanical failure classification models. Saves best model artifact to `artifacts/MechanicalFailure/`. |
| 3 | `02_delta_models.ipynb` | Trains regression models for each delta alignment parameter (LL, SS, L4S1, GlobalTilt, T4PA, L1PA, SVA) and ODI. Saves model artifacts to `artifacts/`. |
| 4 | `03_optimization_holdout_patients.ipynb` | Loads all trained models, runs the GA optimization for multiple scenarios on holdout patients, and displays results. |
| 5 | `04_optimization_new_patient.ipynb` | Accepts manually entered preop data or CSV input (single or multiple rows), validates inputs, computes preop GAP, and runs optimization. |

---

## Running the Optimization Notebook (`03_optimization_holdout_patients.ipynb`)

For a patient not present in your holdout CSV, use `04_optimization_new_patient.ipynb`.

## New Patient Input (`04_optimization_new_patient.ipynb`)

Recommended file location for CSV input:

- `data/raw/new_patient_input_template.csv`

The notebook supports either:

- **Dict mode**: edit one Python dictionary directly in the notebook
- **CSV mode**: add one or more patient rows to the CSV template and set `INPUT_MODE = "csv"`

### Accepted values

- `sex`: `MALE` / `FEMALE` (also accepts `M` / `F`)
- `revision`, `smoking`: `0` or `1`
- `ASA_CLASS`: `1` to `5`
- `id`: optional (recommended for multi-patient CSV files)

### Required columns

Use the template at `data/raw/new_patient_input_template.csv`.

- Source of truth for required model fields: `config.PATIENT_FIXED_COLS`
- Additional required field for mech-failure prediction: `smoking`

To change accepted categorical/binary values, update `src/optimization_utils.py`:
- `_ALLOWED_INPUT_VALUES`

### 1. GA Configuration

The configuration cell at the top of the notebook controls all tunable parameters:

```python
POP_SIZE  = 100    # GA population size
N_GEN     = 5      # Number of generations
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

### 3. Patient Selection

The notebook automatically loops over all holdout patients from `data/processed/holdout_patients.csv`.

To change the holdout set, re-run `00_data_cleaning.ipynb` (which uses PCA + farthest-point sampling) or edit `N_HOLDOUT` in `src/config.py`.

### 4. `DEDUP_BEST` — Deduplicate Best Plans

When `DEDUP_BEST = True`, the "Best Solution per Scenario" table ensures each scenario displays a distinct surgical plan. If two scenarios both select the same best plan, the second one will fall back to its second best diverse alternative, so you can see a wider range of recommendations

Set `DEDUP_BEST = False` to allow the same plan to appear as "best" in multiple scenarios.

### 5. `PSO_LL_OVERRIDE` — Delta LL Correction

When `PSO_LL_OVERRIDE = True`, if the surgical plan includes a PSO/osteotomy, the predicted delta LL is clamped to the clinically expected range of **20–45°**.