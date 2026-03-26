# 🧮 AI Math Olympiad Solver

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Production-ready ML system for classifying and solving Math Olympiad-level problems using multi-model inference.**

[Overview](#-overview) • [Architecture](#-architecture) • [Features](#-features) • [Quick Start](#-quick-start) • [Pipeline](#-ml-pipeline) • [API](#-api-deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [ML Pipeline](#-ml-pipeline)
- [API Deployment](#-api-deployment)

---

## 🧩 Overview

The **AI Math Olympiad Solver** is a production-grade machine learning system designed to classify complex mathematical problems. Unlike simple notebook projects, this system implements a clean, modular, and automated ML pipeline capable of training both traditional ML models and deep neural networks to achieve state-of-the-art classification performance.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Math Dataset│───▶│ Data Ingestion│───▶│   Preprocessing  │  │
│  │  (CSV/Text)  │    │  (Validation) │    │  (TF-IDF/Seq)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                   │             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  API Endpoint│◀───│   Inference  │◀───│   Model Trainer  │  │
│  │  (FastAPI)   │    │   Pipeline   │    │  (RF + PyTorch)  │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                   │             │
│                      ┌────────────────────────────┘             │
│                      ▼                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                Automated Evaluation                        │  │
│  │    (Metrics: Acc, Prec, Recall, F1 + Visualizations)       │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Dual-Model Approach** | Compare performance between Random Forest and PyTorch Neural Networks |
| 🔄 **Modular Pipeline** | Fully automated ingestion, preprocessing, training, and evaluation |
| 🌐 **FastAPI Backend** | Production-ready REST API with Swagger documentation |
| 📉 **Automated Monitoring** | Automatic generation of confusion matrices and loss curves |
| ⚙️ **Config-Driven** | Manage experiment hyper-parameters via `config.yaml` |
| 🛠️ **Robustness** | Custom logging and centralized error handling |

---

## 📁 Project Structure

```
AI_Math_Olympiad_Solver/
│
├── app/                  # FastAPI Application
│   ├── main.py           # Application factory
│   ├── router.py         # API route handlers
│   └── schemas.py        # Pydantic validation models
│
├── config/
│   └── config.yaml       # Central configuration
│
├── data/                 # Raw datasets
│
├── src/                  # Core ML Pipeline
│   ├── components/       # Ingestion, Preprocessing, Trainer, Evaluation
│   ├── models/           # PyTorch architecture
│   ├── pipeline/         # Training & Inference orchestrators
│   └── utils/            # Shared helpers
│
├── artifacts/            # Generated models, reports, plots
├── run.py                # Convenience server runner
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/Vanshika0912/AI_Math_Olympiad_Solver.git
cd AI_Math_Olympiad_Solver

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Training Pipeline

Run the end-to-end training orchestrator:

```bash
python -m src.pipeline.training_pipeline
```

This will:
1. Load and clean the dataset.
2. Train both traditional ML and Deep Learning models.
3. Compute metrics and plot performance.
4. Automatically persist the best model to `artifacts/models/`.

### 3. API Deployment

Start the API:

```bash
python run.py
```

Access API documentation at `http://localhost:8000/docs`.

---

## 🤖 ML Pipeline

### Training Workflow
1. **Data Ingestion**: Schema validation and train/test split (stratified).
2. **Preprocessing**: Text cleaning (regex), TF-IDF vectorization, and sequence mapping for DL.
3. **Training**: Concurrent training of `RandomForestClassifier` and custom PyTorch `MathProblemClassifier`.
4. **Evaluation**: Automated metrics computation, confusion matrix plotting, and model selection.

---

## 🌐 API Deployment

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Check service & model status |
| `POST` | `/api/v1/predict` | Classify math problem |
| `POST` | `/api/v1/train` | Trigger re-training |

### Example Prediction
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"problem": "Prove that for any integer n, n^3 - n is divisible by 6."}'
```
