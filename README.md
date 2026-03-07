# Supply Chain Order Delay Prediction Engine
### **Production MLOps Pipeline: Snowflake ↔️ MLflow ↔️ FastAPI ↔️ Docker**

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![MLOps](https://img.shields.io/badge/MLOps-Production--Ready-green.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-009688.svg)

## Project Vision
This repository contains a production-grade machine learning system designed to predict **Late Delivery Risks** in global logistics. Unlike a simple model script, this is a **modular ecosystem** built for reliability, scalability, and auditability. It bridges the gap between a cloud data warehouse (**Snowflake**) and a real-time inference service (**FastAPI**), protected by a **Statistical Drift Audit** layer.



---

##  System Architecture
The pipeline is strictly decoupled into functional layers to ensure high maintainability:

1.  **Data Ingestion Layer**: Robust loading from CSV/Snowflake with automated encoding detection (UTF-8/Latin-1) and data integrity logging.
2.  **Feature Store (Snowflake)**: A dedicated interface to sync engineered features back to a Snowflake Feature Table, ensuring a "Single Source of Truth" across environments.
3.  **Automated Feature Engineering**: 
    * **Temporal:** Cyclical encoding of order hours and days.
    * **Geospatial:** Cross-country shipping flags.
    * **Selection:** Automated removal of zero-variance and low-correlation (corr < 0.01) features.
4.  **Model Tournament (MLflow)**: Runs a competitive training session between **Logistic Regression**, **Random Forest**, and **HistGradientBoosting**. The "Champion" is automatically serialized based on ROC-AUC/F1-score.
5.  **Inference Layer**: A containerized FastAPI service with strict Pydantic schema validation for high-performance serving.
6.  **Monitoring Layer**: A statistical audit suite using **Kolmogorov-Smirnov** and **Chi-Square** tests to detect feature drift before retraining cycles.



---

##  Repository Structure
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
