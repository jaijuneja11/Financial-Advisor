# 💸 AI-Powered Financial Advisor Web App

An intelligent Financial Advisor application that uses **Machine Learning models** and **Google's Gemini AI API** to assist users in:

- Predicting personalized savings rates
- Detecting overspending habits
- Interacting with a conversational Gemini-powered chatbot for financial guidance

Built using **Flask**, **scikit-learn**, **LightGBM**, **Plotly**, and **Gemini API**.

---

## 🚀 Features

- 🔮 Predict monthly **Savings Rate** based on your financial profile
- ⚠️ Get warnings on potential **Overspending Behavior**
- 📊 Interactive **Visualizations**: Pie chart, Bar chart, Treemap, and Gauge
- 🤖 Chat with an **AI-powered Gemini Bot** for real-time financial queries

---

## 🧠 How It Works

### 🔍 ML Models Used

- **Savings Rate Predictor**  
  Model: Random Forest  
  Inputs:  
  - Income, Age, Dependents, Occupation  
  - Desired Savings %, Expense Ratios  
  Output: Predicted monthly savings rate (%)

- **Overspending Classifier**  
  Model: LightGBM  
  Inputs:  
  - Lifestyle spend breakdown (groceries, entertainment, eating out, etc.)  
  Output: Binary classifier indicating overspending risk

### 📊 Visualization Tools

- **Bar Chart** – Spending per category  
- **Pie Chart** – Proportional spending  
- **Treemap** – Expense hierarchy  
- **Gauge** – Compare actual vs target savings

### 🤖 Gemini AI Chatbot

- Powered by Google's **Gemini 1.5 Flash model** via REST API  
- Understands natural language financial questions  
- Provides dynamic, AI-generated financial suggestions

---

## 📁 Project Structure

├── app.py # Flask backend logic and routes

├── templates/
│ ├── index.html # Form interface

  └── result.html # Result and chart page

├── savings_rate_model_rf_optimized.pkl # Random Forest savings model

├── lightgbm_overspending_model.pkl # LightGBM overspending classifier

├── requirements.txt # All required dependencies

yaml
Copy
Edit

---

## ⚙️ Installation & Setup

### 🔧 Prerequisites

- Python 3.8+
- pip package manager
