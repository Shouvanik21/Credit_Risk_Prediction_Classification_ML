# 💳 CreditLens — Credit Risk Prediction Using Machine Learning

## 📌 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Model Performance](#-model-performance)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Application Workflow](#-application-workflow)
- [Installation and Execution](#-installation-and-execution)
- [Future Improvements](#-future-improvements)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

**CreditLens** is a Machine Learning-based Credit Risk Prediction application designed to identify whether a loan applicant is likely to default on a loan based on their financial details and credit history.

The project explores multiple classification algorithms, compares their performance, and selects a Support Vector Machine (SVM) model based on the evaluation results.

The selected model is integrated into an interactive **Streamlit web application**, allowing users to enter applicant information and receive a credit risk prediction along with the model's prediction confidence.

## ❗ Problem Statement

Financial institutions need to assess the possibility of loan defaults before making lending decisions. Identifying applicants who may fail to repay their loans can help lenders better understand potential credit risk.

The objective of CreditLens is to build a **binary classification system** that uses historical borrower and loan information to predict the likelihood of loan default.

## ✨ Key Features

| Feature | Description |
|---|---|
| 🎯 Credit Default Prediction | Classifies applicants as defaulters or non-defaulters. |
| 🤖 Multiple Model Comparison | Evaluates five different classification algorithms. |
| 📊 Model Evaluation | Compares models using Accuracy and F1 Score. |
| 🔍 Prediction Confidence | Displays the model's confidence associated with its prediction. |
| 🌐 Interactive Web Application | Provides an easy-to-use Streamlit interface. |
| 💾 Model Persistence | Saves the trained model and preprocessing artifacts using Pickle. |

## 📂 Dataset

**Dataset:** Credit Risk Dataset  
**Source:** [Kaggle — Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

The dataset contains **32,581 records and 12 columns**, covering borrower demographics, financial information, loan details, and credit history.

### Target Variable

The target column is `loan_status`.

| Value | Meaning |
|:---:|---|
| `0` | No default |
| `1` | Default |

### Dataset Features

| Feature | Description |
|---|---|
| `person_age` | Age of the loan applicant |
| `person_income` | Applicant's annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Assigned loan grade |
| `loan_amnt` | Requested loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a proportion of annual income |
| `cb_person_default_on_file` | Historical default indicator |
| `cb_person_cred_hist_length` | Length of credit history in years |
| `loan_status` | Target variable indicating loan default |

## 🧠 Machine Learning Models

Five classification algorithms were implemented and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Naive Bayes
4. Decision Tree
5. Support Vector Machine (SVM)

### 📈 Model Performance

The models were compared using two evaluation metrics:

- **Accuracy:** Measures the proportion of correct predictions.
- **F1 Score:** Balances Precision and Recall, providing a useful measure when class distributions are imbalanced.

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 86.58% | 64.53% |
| K-Nearest Neighbors | 89.52% | 72.26% |
| Naive Bayes | 83.34% | 55.47% |
| Decision Tree | 88.43% | 74.37% |
| **Support Vector Machine** | **91.27%** | **76.72%** |

### 🏆 Selected Model — Support Vector Machine

Based on the evaluated results, **SVM achieved the highest Accuracy and F1 Score** among the five tested algorithms.

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-91.27%25-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/F1%20Score-76.72%25-blue?style=for-the-badge" />
</p>

The SVM model was selected as the final model and saved for integration into the Streamlit application.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Model training and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Jupyter Notebook | Model development and experimentation |
| Streamlit | Interactive web application |
| Pickle | Model and preprocessing artifact persistence |

## 📁 Project Structure

```text
CreditLens/
│
├── credit_risk_dataset.csv
├── credit_risk_prediction.ipynb
├── README.md
│
├── app.py
├── requirements.txt
├── start.sh
│
├── svm_CreditRisk_model.pkl
├── scaler_CreditRisk.pkl
└── feature_columns_CreditRisk.pkl
```

## 🔄 Application Workflow

             Dataset Loading
                    │
                    ▼
     Exploratory Data Analysis (EDA)
                    │
                    ▼
       Data Quality Assessment
       (Null Values & Duplicates)
                    │
                    ▼
       Data Cleaning & Preprocessing
       (Missing Values, Duplicates,
             One-Hot Encoding)
                    │
                    ▼
            Feature Engineering
                    │
                    ▼
             Feature Selection
          (Features and Target)
                    │
                    ▼
              Train-Test Split
                    │
                    ▼
             Feature Standardization
                    │
                    ▼
          Model Training & Evaluation
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     Five ML Models       Performance
                           Comparison
          └─────────┬─────────┘
                    │
                    ▼
          Best Model Selection
          (Accuracy & F1 Score)
                    │
                    ▼
          Model & Artifacts Saved
                    │
                    ▼
          Streamlit Frontend
                    │
                    ▼
         Credit Risk Prediction
                    │
                    ▼
          Prediction & Confidence

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd Credit_Risk_Prediction_Classification_ML
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Launch the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

## ⚠️ Disclaimer

CreditLens is developed strictly for **educational and machine learning experimentation purposes**. The predictions generated by this application should not be considered professional financial advice or used as the sole basis for real-world lending decisions.

---

## 📝 Conclusion

CreditLens demonstrates an end-to-end Machine Learning workflow for credit risk prediction, covering data exploration, cleaning, preprocessing, feature engineering, model training, and evaluation.

By comparing five classification algorithms, the project identified Support Vector Machine (SVM) as the best-performing model based on Accuracy and F1 Score. The selected model was then integrated into a Streamlit application to make credit risk predictions through a simple interactive interface.

This project provided practical experience in applying Machine Learning techniques to a real-world financial problem and transforming a trained model into a usable application.