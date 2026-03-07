# 🚛 Supply Chain Order Delay Prediction Engine
### **Production MLOps Pipeline: Snowflake ↔️ MLflow ↔️ FastAPI ↔️ Docker**

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-green.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg)

## 📖 Project Vision
This repository contains a production-grade machine learning system designed to predict **Late Delivery Risks** in global logistics. Unlike a simple model script, this is a **modular ecosystem** built for reliability, scalability, and auditability. It bridges the gap between a cloud data warehouse (**Snowflake**) and a real-time inference service (**FastAPI**), protected by a **Statistical Drift Audit** layer.



---

## 🏗️ Detailed System Architecture & Design Patterns

### 1. Robust Data Ingestion & Cleaning
Production data is rarely clean. This pipeline implements an **Encoding-Aware Ingestion** strategy:
* **Fail-Safe Loading:** Automatically attempts `UTF-8` and falls back to `Latin-1` to prevent pipeline crashes during automated runs.
* **Config-Driven Cleaning:** Instead of hard-coding logic, all NA-filling (`Sales: 0`, `Quantity: 1`) and type conversions are handled via `config.yaml`. This allows the business logic to change without touching the source code.

### 2. Feature Store & Engineering Strategy
To ensure consistency between training and serving (preventing **Training-Serving Skew**):
* **Snowflake Integration:** Features are synced back to Snowflake. This allows other teams to consume the same "Source of Truth" features for BI or other models.
* **Geospatial & Temporal Logic:** We engineer high-signal features like `cross_country_flag` (comparing Customer vs. Order country) and cyclical `order_hour` extraction.
* **Automated Pruning:** A correlation-based selector removes features with a coefficient $< 0.01$, reducing model noise and improving training speed.



### 3. The "Model Tournament" (Auto-Selection)
Rather than assuming one algorithm is best, `modeling.py` implements a competitive selection process:
* **MLflow Tracking:** Every experiment logs hyperparameters, F1-scores, and ROC-AUC curves. 
* **Candidate Models:** We compare **Logistic Regression** (baseline), **Random Forest** (non-linear), and **HistGradientBoosting** (high-performance boosting).
* **Artifact Persistence:** The winning "Champion" is automatically promoted to the `artifacts/` folder, ready for the API to load.

### 4. Enterprise Logging & Observability
* **Log Rotation:** The custom `logger.py` implements a retention policy, keeping only the last 10 logs per component. This prevents the server from running out of disk space during high-frequency retraining.
* **Unified Entry Point:** `main.py` acts as the orchestrator, ensuring that data flows sequentially from Ingestion → Cleaning → Engineering → Modeling.

---

## 📂 Repository Structure
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
