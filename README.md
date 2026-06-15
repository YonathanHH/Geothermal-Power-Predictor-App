## 📌 Overview

This repository contains the **Geothermal Power Output Predictor**, a Streamlit application that estimates geothermal well power generation (MWe) from reservoir and well properties. The project is based on a physics-informed synthetic geothermal dataset and includes both the deployment app and the Jupyter notebook workflow used for data preparation, model training, and evaluation.

Find my dataset here **[Dataset Repositry](https://github.com/YonathanHH/Synthetic-Geothermal-Dataset)**

<img width="1912" height="882" alt="image" src="https://github.com/user-attachments/assets/ecd8b4ea-b24c-473c-b5ea-cb85682a2745" />

## ✨ Highlights

- Interactive **Streamlit app** for predicting geothermal power output.
- **Map-based location selector** with automatic latitude, longitude, and country detection.
- Random Forest regression model trained with `GridSearchCV`.
- SHAP-based feature interpretation for model explainability.
- Jupyter notebooks included for reproducibility and exploration.
- Batch prediction support through CSV upload.

## 📁 Repository Contents

- `app.py` — Streamlit application.
- `train_and_save.py` — Model training and artifact saving script.
- `requirements.txt` — Python dependencies.
- `notebooks/geothermal_power_output_prediction.ipynb` — Jupyter notebooks for feature engineering, model training, and evaluation.
- `model.joblib` — Trained prediction model.
- `label_encoder.joblib` — Country encoder used by the app.

## 🚀 Demo

You can try the deployed app here: **[Streamlit App](https://geothermalpp.streamlit.app/)**

## 📓 Notebooks

The notebook files in this repository document the full workflow:
1. dataset loading,
2. feature engineering,
3. model training,
4. evaluation metrics.

You can open the notebooks directly in GitHub or run them locally in Jupyter Notebook / JupyterLab.

## 🛠️ Local Setup

```bash
git clone https://github.com/YonathanHH/Geothermal-Power-Predictor.git
cd Geothermal-Power-Predictor
pip install -r requirements.txt
python train_and_save.py
streamlit run app.py
```

## 📊 Model Summary

- **Target:** `power_generated_MWe`
- **Model:** Random Forest Regressor
- **Tuning:** `GridSearchCV`
- **Input features:** depth, temperature, pressure, porosity, permeability, flow rate, location, country, and reservoir type

## 🔎 Notes

This project uses a synthetic dataset designed to be physically realistic for educational and portfolio purposes. It is not intended for operational geothermal field design.

## 👤 Author

YonathanHH  
GitHub: [YonathanHH](https://github.com/YonathanHH)
