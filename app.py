from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import requests
import os

app = Flask(__name__)

# ─── Load Models ─────────────────────────────
savings_model = joblib.load("savings_rate_model_rf_optimized.pkl")
overspend_model = joblib.load("lightgbm_overspending_model.pkl")

# ─── Gemini REST API Setup ───────────────────
GEMINI_API_KEY = "AIzaSyDq1Q8tEgdDIogb5P--TSGLVCzb1UeGBdE"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
HEADERS = {
    "Content-Type": "application/json",
    "x-goog-api-key": GEMINI_API_KEY
}

# ─── Routes ──────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    income = float(form["income"])
    age = int(form["age"])
    dependents = int(form["dependents"])
    city_encoded = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}[form["city_tier"]]
    desired_pct = float(form["desired_savings_pct"]) / 100

    occupation_map = {
        "Professional": 0,
        "Retired": 1,
        "Self_Employed": 2,
        "Student": 3
    }
    occupation = occupation_map[form["occupation"]]

    # Expenses
    rent = float(form["rent"])
    insurance = float(form["insurance"])
    transport = float(form["transport"])
    education = float(form["education"])
    loan = float(form["loan"])
    groceries = float(form["groceries"])
    eating_out = float(form["eating_out"])
    entertainment = float(form["entertainment"])
    utilities = float(form["utilities"])
    misc = float(form["miscellaneous"])

    total_spend = sum([
        rent, insurance, transport, education, loan,
        groceries, eating_out, entertainment, utilities, misc
    ])
    disposable = income - total_spend

    # Model Inputs
    savings_input = pd.DataFrame([{
        "Income": income,
        "Age": age,
        "Dependents": dependents,
        "Occupation_Encoded": occupation,
        "Desired_Savings_Percentage": desired_pct,
        "pct_rent": rent / income,
        "pct_insurance": insurance / income,
        "pct_transport": transport / income,
        "pct_education": education / income,
        "Has_Loan": int(loan > 0),
    }])

    overspend_input = pd.DataFrame([{
        "Income": income,
        "Age": age,
        "Dependents": dependents,
        "Occupation_Encoded": occupation,
        "City_Tier_Encoded": city_encoded,
        "pct_groceries": groceries / income,
        "pct_eating_out": eating_out / income,
        "pct_entertainment": entertainment / income,
        "pct_miscellaneous": misc / income,
        "pct_utilities": utilities / income
    }])

    # Predictions
    savings_rate = savings_model.predict(savings_input)[0] * 100
    overspend = overspend_model.predict(overspend_input)[0]
    gap = (desired_pct * 100) - savings_rate

    # Charts
    labels = [
        "Rent", "Loan", "Insurance", "Groceries", "Transport",
        "Eating Out", "Entertainment", "Utilities", "Education",
        "Miscellaneous", "Savings"
    ]
    values = [
        rent, loan, insurance, groceries, transport,
        eating_out, entertainment, utilities, education,
        misc, max(0, disposable)
    ]
    df = pd.DataFrame({"Category": labels, "Amount": values})

    bar = plot(px.bar(df, x="Category", y="Amount", title="Spending per Category (₹)"), output_type="div")
    pie = plot(px.pie(df, names="Category", values="Amount", title="Spending Proportion"), output_type="div")
    tree = plot(px.treemap(df, path=["Category"], values="Amount", title="Spending TreeMap"), output_type="div")
    gauge = plot(go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=savings_rate,
        delta={"reference": desired_pct * 100},
        gauge={"axis": {"range": [0, 100]},
               "bar": {"color": "green"},
               "steps": [
                   {"range": [0, 30], "color": "red"},
                   {"range": [30, 60], "color": "orange"},
                   {"range": [60, 100], "color": "lightgreen"}]},
        title={"text": "Savings vs Target"})), output_type="div")

    return render_template("result.html", savings_rate=round(savings_rate, 2),
                           savings_gap=round(gap, 2), overspend=overspend,
                           bar_chart=bar, pie_chart=pie, treemap=tree, gauge_chart=gauge)

# ─── Gemini Chat via REST API ─────────────────
@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"reply": "🙏 Please say something first."})

    payload = {
        "contents": [
            {
                "parts": [{"text": user_msg}]
            }
        ]
    }

    try:
        res = requests.post(GEMINI_API_URL, headers=HEADERS, json=payload)
        data = res.json()
        reply = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "🤖 No reply.")
    except Exception as e:
        reply = f"⚠️ Gemini Error: {str(e)}"

    return jsonify({"reply": reply})

# ─── Run App ──────────────────────────────────
# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
