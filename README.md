# ⚡ Algorithmic Pricing Engine for Short-Term Rentals

An interactive, end-to-end Machine Learning web application designed to optimize pricing strategies for short-term rental properties (e.g., Airbnb). It moves beyond basic price prediction to **prescriptive analytics**, recommending the exact price that maximizes expected revenue based on specific market conditions.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33.0-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-F7931E)

## 📌 The Business Problem
Property managers often rely on static pricing or simple rules (e.g., +20% on weekends). This leaves money on the table. By understanding **Price Elasticity of Demand**—how booking probability changes as price changes under different conditions (weather, holidays, local events)—we can algorithmically determine the optimal price point that maximizes Expected Revenue.

## 🚀 Features
- **Synthetic Data Pipeline**: A robust `data_generator.py` script that models complex real-world dynamics, generating 365 days of synthetic booking data for 50 properties, factoring in seasonality, weather impacts, and weekend/holiday surges.
- **Machine Learning Model**: A Scikit-Learn `RandomForestClassifier` trained to predict the probability of a booking given a specific price and context.
- **Revenue Optimization Engine**: Calculates the Expected Revenue curve (Price × Booking Probability) to find the mathematical peak (the optimal price).
- **Interactive UI**: A sleek, dark-themed Streamlit dashboard with glassmorphism UI elements, dynamic Plotly visualizations, and model explainability (Feature Importance).

## 🛠️ Tech Stack
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (Random Forest)
- **Web Framework**: `streamlit`
- **Visualizations**: `plotly`

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-link>
   cd algorithmic_pricing_engine
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* **Re-generate data and re-train the model:**
   The repository comes with a pre-trained model and data, but you can generate your own:
   ```bash
   python data_generator.py
   python train_model.py
   ```

4. **Launch the Dashboard:**
   ```bash
   streamlit run app.py
   ```

## 🖥️ DashBoard Preview
   <video src="
https://github.com/user-attachments/assets/81a4a6c0-54e3-4106-9ee8-024d96733133
" width="100%"></video>


## 🧠 Methodology & Analytics Approach
1. **Feature Engineering**: Encoded categorical variables (neighborhood, weather) and boolean flags (is_weekend, is_holiday) to capture the market context.
2. **Probability Calibration**: The Random Forest outputs probability scores (`predict_proba`). We test a range of prices (from 40% to 280% of the base price) through the model to map out the entire Demand Curve.
3. **Expected Value**: We optimize for `max(Price * P(Booking))`, not just maximum occupancy.
