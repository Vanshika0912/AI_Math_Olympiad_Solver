# 🧠 AI Math Olympiad Solver – End-to-End ML Pipeline

<p align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch" />
<img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn" />
<img src="https://img.shields.io/badge/FastAPI-Production-green?logo=fastapi" />
<img src="https://img.shields.io/badge/Machine-Learning-purple" />
<img src="https://img.shields.io/badge/End--to--End-ML%20Pipeline-blueviolet" />
<img src="https://img.shields.io/badge/NLP-Text%20Processing-yellow" />
<img src="https://img.shields.io/badge/Docker-Ready-blue?logo=docker" />
<img src="https://img.shields.io/badge/License-MIT-lightgrey" />

</p>

<p align="center">
<strong>Production-ready AI system that solves math olympiad-level problems using a modular ML pipeline and real-time inference API.</strong>
</p>

<p align="center">
Overview • Architecture • Features • Pipeline • Training • Inference • API • Results
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [ML Pipeline](#-ml-pipeline)
- [Training Pipeline](#-training-pipeline)
- [Inference Pipeline](#-inference-pipeline)
- [FastAPI Deployment](#-fastapi-deployment)
- [Model Evaluation](#-model-evaluation)
- [Results & Performance](#-results--performance)
- [API Reference](#-api-reference)
- [Future Improvements](#-future-improvements)

---

## 🧠 Project Overview

This project builds a **production-grade AI system capable of solving math olympiad-level problems** using a structured machine learning pipeline.

Unlike traditional math solvers that rely purely on symbolic logic, this system focuses on:

- Learning patterns from mathematical problems  
- Automating the full ML lifecycle  
- Providing real-time predictions via API  
- Supporting retraining and continuous improvement  

The goal is to demonstrate how **ML systems are actually built in production**, not just trained in notebooks.

---

## 🏗️ System Architecture
Math Problem Dataset
│
▼
Text Cleaning & Preprocessing
│
▼
Feature Engineering (TF-IDF / Vectorization)
│
▼
Model Training
├── Traditional ML Models (SVM, Random Forest, Logistic Regression)
└── Deep Learning Model (PyTorch Neural Network)
│
▼
Model Evaluation & Selection
│
▼
Saved Model + Preprocessing Pipeline
│
▼
Inference Pipeline
│
▼
FastAPI REST API (/predict)


---

## ✨ Features

- Solves olympiad-level math problems
- End-to-end ML pipeline (data → training → deployment)
- Modular training pipeline for reproducibility
- Separate inference pipeline for real-time predictions
- Traditional ML + Deep Learning models
- Automatic best model selection
- Production-ready FastAPI API
- Logging and exception handling
- Docker-ready deployment
- Clean, modular, production-style code

---

## 🛠️ Tech Stack

**Programming:** Python 3.10+  
**Machine Learning:** Scikit-learn  
**Deep Learning:** PyTorch  
**Data Processing:** Pandas, NumPy  
**Feature Engineering:** TF-IDF / Text Vectorization  
**API:** FastAPI + Uvicorn  
**Visualization:** Matplotlib, Seaborn  
**Deployment:** Docker  
**Architecture:** Modular ML pipelines  

---

## ⚙️ ML Pipeline

The project follows a real production ML workflow:


Raw Data → Data Cleaning → Text Processing → Feature Engineering
→ Model Training → Model Evaluation → Best Model Selection
→ Model Saving → Inference Pipeline → FastAPI Deployment


---

## 🧪 Training Pipeline

The training pipeline includes:

- Data ingestion
- Text preprocessing
- Feature extraction (TF-IDF)
- Training multiple ML models
- Training a PyTorch neural network
- Performance comparison
- Automatic best model selection
- Saving model + preprocessing pipeline

This makes the system **reproducible and scalable**.

---

## ⚡ Inference Pipeline

The inference pipeline is fully separated from training and includes:

1. Loading the trained model
2. Applying the same preprocessing steps
3. Converting input math problems into features
4. Generating predictions in real time

This ensures **consistent and production-safe predictions**.

---

## 🌐 FastAPI Deployment

The project includes a production-ready FastAPI service.

### POST /predict

**Input**
```json
{
  "problem": "Find the value of x if 2x + 5 = 15"
}

Output

{
  "prediction": "x = 5"
}
GET /health

Used to check whether the API is running.

📊 Model Evaluation

The system evaluates models using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
Model comparison reports

The best-performing model is automatically selected and saved.

📈 Results & Performance

The system is designed to:

Learn mathematical patterns from training data
Handle structured math questions
Provide fast real-time predictions
Support continuous retraining

Performance improves as more olympiad-level datasets are added.

🚀 Why This Project Matters

This project demonstrates real skills required for:

Machine Learning Engineer roles
AI Engineer roles
Production ML roles
End-to-End ML system design

Instead of just training a model, this project focuses on building a production-ready AI application.

🔮 Future Improvements
Transformer-based models (BERT / LLM fine-tuning)
Step-by-step solution generation
Symbolic math solving integration
Math expression parsing (LaTeX support)
Web interface for students
Auto-dataset expansion
⭐ If You Like This Project

Give this repository a ⭐ on GitHub and connect with me if you're interested in collaborating on AI/ML or MLOps projects.


---
