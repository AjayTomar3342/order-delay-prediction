#  Supply Chain Order Delay Prediction Engine
### **Production MLOps Pipeline: Snowflake ↔️ MLflow ↔️ FastAPI ↔️ Docker**

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-green.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg)

## 📖 Project Vision
This repository contains a production-grade machine learning system designed to predict **Late Delivery Risks** in global logistics. Unlike a simple model script, this is a **modular ecosystem** built for reliability, scalability, and auditability. It bridges the gap between a cloud data warehouse (**Snowflake**) and a real-time inference service (**FastAPI**), protected by a **Statistical Drift Audit** layer.

---

## Detailed System Architecture & Design Patterns

### 1. Robust Data Ingestion & Cleaning
Production data is rarely clean. This pipeline implements an **Encoding-Aware Ingestion** strategy:

- **Fail-Safe Loading:** Automatically attempts `UTF-8` and falls back to `Latin-1` to prevent pipeline crashes during automated runs.
- **Config-Driven Cleaning:** Instead of hard-coding logic, all NA-filling (`Sales: 0`, `Quantity: 1`) and type conversions are handled via `config.yaml`. This allows the business logic to change without touching the source code.

### 2. Feature Store & Engineering Strategy
To ensure consistency between training and serving (preventing **Training-Serving Skew**):

- **Snowflake Integration:** Features are synced back to Snowflake. This allows other teams to consume the same "Source of Truth" features for BI or other models.
- **Geospatial & Temporal Logic:** We engineer high-signal features like `cross_country_flag` (comparing Customer vs. Order country) and cyclical `order_hour` extraction.
- **Automated Pruning:** A correlation-based selector removes features with a coefficient `< 0.01`, reducing model noise and improving training speed.

### 3. The "Model Tournament" (Auto-Selection)
Rather than assuming one algorithm is best, `modeling.py` implements a competitive selection process:

- **MLflow Tracking:** Every experiment logs hyperparameters, F1-scores, and ROC-AUC curves.
- **Candidate Models:** We compare **Logistic Regression** (baseline), **Random Forest** (non-linear), and **HistGradientBoosting** (high-performance boosting).
- **Artifact Persistence:** The winning "Champion" is automatically promoted to the `artifacts/` folder, ready for the API to load.

### 4. Enterprise Logging & Observability

- **Log Rotation:** The custom `logger.py` implements a retention policy, keeping only the last 10 logs per component. This prevents the server from running out of disk space during high-frequency retraining.
- **Unified Entry Point:** `main.py` acts as the orchestrator, ensuring that data flows sequentially from Ingestion → Cleaning → Engineering → Modeling.

---

## Repository Structure

```bash
.
├── .github/workflows/       # CI/CD: Automated Retraining & Deployment
├── config/                  # Centralized YAML configurations (brain of the project)
├── src/
│   ├── api/                 # FastAPI Implementation (App, Schema, Predictor)
│   ├── ingestion/           # Snowflake & CSV Ingestion Logic
│   ├── preparation/         # Cleaning & Feature Engineering Classes
│   ├── modeling/            # Tournament Trainer & MLflow Tracking
│   ├── monitoring/          # Statistical Drift Detection (KS-Test/Chi2)
│   └── utils/               # Config Loaders & Rotation Loggers (last 10 runs)
├── artifacts/               # Serialized Champion Models (.pkl)
├── logs/                    # Execution logs (Automated rotation)
├── monitoringresults/       # JSON Drift Reports
└── Dockerfile               # Production Containerization
```

---

##  Execution Guide & Workflow

This project is designed as a linear MLOps pipeline. Each stage generates artifacts (logs, processed data, or models) required by the next stage.

### 1. Environment Setup

To ensure reproducibility, install the exact versions of the dependencies used during development.

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Phase 1: The Training Pipeline

Running the orchestrator triggers the full data-to-model lifecycle. This script automates ingestion, cleaning, feature engineering, and the "Model Tournament."

```bash
python main.py
```

Pipeline Actions:

- **Ingestion:** Loads `shipments_raw.csv` with automatic encoding fallback.
- **Cleaning:** Implements config-driven NA filling and type casting.
- **Engineering:** Extracts temporal features and performs correlation-based pruning.

---

### Phase 2: Experiment Tracking (MLflow)

Every run is tracked. You can compare the performance of Logistic Regression, Random Forest, and Gradient Boosting side-by-side.

```bash
mlflow ui
```

View metrics at:

```
http://127.0.0.1:5000
```

---

### Phase 3: Data Drift Audit (Monitoring)

Before deploying, or as a scheduled health check, run the drift audit. This compares the Training Baseline against Current Data.

```bash
python src/monitoring/monitor_drift.py
```

Statistical Tests Performed:

- **Numerical Features:** Kolmogorov-Smirnov (KS) Test (`p < 0.05`)
- **Categorical Features:** Chi-Square Contingency Test (`p < 0.05`)

---

### Phase 4: Production API Serving (FastAPI)

Deploy the winning model as a REST API. The service includes a predictor class that handles the mapping of API inputs to the model's expected feature names.

```bash
uvicorn src.api.app:app --reload
```

---

### Phase 5: Containerization (Docker)

Package the entire environment (OS, Python, and Libraries) to ensure "run anywhere" stability.

```bash
# Build the image
docker build -t order-delay-api .

# Run the container
docker run -d -p 8000:8000 --name shipment-service order-delay-api
```

---

## Monitoring & Governance

To prevent **Silent Model Decay**, this system uses automated statistical guards:

- **Numerical Drift:** Compares the cumulative distribution of live features against the training baseline.
- **Categorical Drift:** Ensures the proportions of categories (e.g., Market, Shipping Mode) haven't shifted significantly.
- **Gatekeeping:** The generated `drift_report.json` serves as a programmatic **CI/CD gate** — if drift is detected, deployment of a new model can automatically be blocked.

---

## Technical "War Stories" & Decisions

**The Encoding Challenge**

Implemented a multi-pass ingestion strategy (`UTF-8` → `Latin-1`) to ensure high pipeline availability despite messy raw data.

**Training-Serving Skew**

Decoupled API design from raw data names using a mapping layer in `predictor.py`, making the system robust to upstream field name changes.

**Memory Management**

Implemented automated log rotation in `logger.py` to ensure long-term stability in production environments by keeping only the 10 most recent runs.
