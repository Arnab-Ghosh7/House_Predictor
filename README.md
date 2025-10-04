# 🏠 Gurgaon House Prices Predictor  

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow.svg)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with ❤️](https://img.shields.io/badge/Made%20with-❤️%20by%20Arnab%20Ghosh-red.svg)]()

---

## 📖 Project Description  

**Gurgaon House Prices Predictor** is a **machine learning project** that aims to predict housing prices in **Gurgaon, India**, one of the fastest-growing real estate markets in the country.  

Using a dataset that includes various property features — such as **area (sq.ft), number of bedrooms (BHK), location, and other attributes** — this model learns complex patterns in the housing data and provides accurate price estimates.  

This project demonstrates the **end-to-end process** of building a data science pipeline:  

- Data collection and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature engineering and encoding  
- Model training, hyperparameter tuning, and evaluation  
- Saving and reusing the trained model for future predictions  

The goal is to assist property buyers, real estate agents, and investors in **estimating fair property prices** in Gurgaon.

---

## 📂 Project Structure  



### 📂 Project Structure

├── input.csv # Main dataset

├── housing.csv # Processed dataset

├── output.csv # Prediction output file

├── main.py # Main training/testing script

├── model.pkl (expected) # Trained model file



---

## 🧠 Key Features  

✅ **Data Cleaning & Preprocessing** – Handles missing values, categorical encoding, and feature scaling.  
✅ **Exploratory Data Analysis (EDA)** – Visualizes trends, correlations, and outliers in Gurgaon’s housing market.  
✅ **Model Training** – Uses Scikit-Learn algorithms such as Linear Regression, Decision Tree, or Random Forest.  
✅ **Model Evaluation** – Calculates R² Score, Mean Squared Error (MSE), and Mean Absolute Error (MAE).  
✅ **Price Prediction** – Provides accurate predictions for new input data.  
✅ **Model Persistence** – Saves the trained model as `model.pkl` for future use without retraining.  

---

## ⚙️ Installation & Setup  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/<your-username>/gurgaon-house-price-predictor.git
cd gurgaon-house-price-predictor
```

# model_usage.yaml

description: |
  Example of how to load the trained model (model.pkl) and make predictions
  using the Gurgaon House Prices Predictor ML model. This demonstrates how
  to use Joblib to load a serialized Scikit-learn model and perform inference
  on new input data.

requirements:
  - python >= 3.10
  - scikit-learn
  - joblib
  - numpy

usage:
  load_model: |
    import joblib

    # Load the trained model
    model = joblib.load('model.pkl')

  example_prediction:
    input_format: "[area, bhk, location_encoded, ...]"
    features: [1500, 3, 2, 1, 0]
    code: |
      predicted_price = model.predict([features])
      print(f"🏡 Predicted Price: ₹{predicted_price[0]:,.2f}")

expected_output:
  predicted_price: "🏡 Predicted Price: ₹12,50,000 (example)"


