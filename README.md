
# 🏦 Loan Approval Prediction System

A machine learning-based web application that predicts whether a loan should be approved based on applicant data. This project is built using Python, Flask, and Scikit-learn and follows a modular ML pipeline design to ensure clean and scalable code.

---

## 📌 Table of Contents

- Project Overview
- Demo
- Features
- Tech Stack
- Installation & Setup
- Running the Project
- Dataset Details
- Model Training & Evaluation
- Contact

---

## Project Overview

This project predicts loan approval status (Yes/No) based on user input like income, loan amount, credit history, etc. It includes:
- Modular pipeline for ingestion, transformation, model training
- Trained classification models
- Flask web app with HTML/CSS frontend
- GridSearchCV for hyperparameter tuning
- Model evaluation and visualization

---

## Demo

Live link : https://loan-approval-prediction-model-production-4265.up.railway.app/  
Localhost example: `http://127.0.0.1:5000`

---

## Features

- User-friendly form-based frontend
- Real-time prediction
- Supports 6 ML algorithms
- Final model optimized with GridSearchCV
- Modular, production-ready pipeline
- Clean code and logs for debugging

---

## Tech Stack

**Frontend:**  
- HTML, CSS, JavaScript, Bootstrap

**Backend:**  
- Python, Flask

**ML & Data Tools:**  
- Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn

**Version Control:**  
- Git, GitHub

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

### 2. Create and Activate a Virtual Environment

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### 1. Train the ML Model

```bash
python src/pipeline/train_pipeline.py
```

This runs:
- Data ingestion (reads and splits the dataset)
- Data transformation (cleans and encodes the data)
- Model training using multiple classifiers
- Final model selection using GridSearchCV

### 2. Start the Web Application

```bash
python app.py
```

Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Dataset Details

The dataset contains 13 features:

| Feature | Description |
|--------|-------------|
| Gender | Male/Female |
| Married | Marital Status |
| Dependents | Number of dependents |
| Education | Graduate/Not Graduate |
| Self_Employed | Yes/No |
| ApplicantIncome | Monthly income of applicant |
| CoapplicantIncome | Monthly income of coapplicant |
| LoanAmount | Amount of loan | (To be entered in Thousands)
| Loan_Amount_Term | Term of loan in months |
| Credit_History | Credit score (0/1) |
| Property_Area | Urban/Rural/Semiurban |
| Loan_Status | Target (Y/N) |

---

## Model Training & Evaluation

Six classification models were trained and compared:
- Logistic Classification
- Random Forest (final model)
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree
- Naive Bayes

**Best Model:** `Random Forest`  
**Tuning:** Performed using `GridSearchCV`  
**Metrics Used:** Accuracy, Precision, Recall, Confusion Matrix

---

## Contact

👨‍💻 **Nishit Kumar**  
🎓 B.Tech, NITK Surathkal  
📧 nishitkumaroll12@gmail.com  
