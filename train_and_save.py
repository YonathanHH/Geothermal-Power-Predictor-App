# ══════════════════════════════════════════════════════════════════════════════
# train_and_save.py — Random Forest Only (Best Model)
# Run once: python train_and_save.py
# Outputs : model.joblib, label_encoder.joblib, training_report.txt
# ══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing   import LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score

print("=" * 60)
print("  GEOTHERMAL POWER PREDICTOR — RANDOM FOREST TRAINING")
print("=" * 60)

# ── 1. Load & Filter ──────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
df   = pd.read_csv("geothermal_wells_worldwide.csv")
prod = df[df["well_type"] == "Production"].copy().reset_index(drop=True)
print(f"      ✓ Production wells loaded: {len(prod):,}")

# ── 2. Feature Engineering ────────────────────────────────────────────────────
print("[2/5] Engineering features...")
prod["log_permeability"] = np.log1p(prod["permeability_mD"])
prod["log_flow_rate"]    = np.log1p(prod["flow_rate_kg_s"])
prod = pd.get_dummies(prod, columns=["field_type"], drop_first=False)

le = LabelEncoder()
prod["country_enc"] = le.fit_transform(prod["country"])

FEATURES = [
    "well_depth_m", "temperature_C", "pressure_MPa", "porosity",
    "log_permeability", "log_flow_rate", "latitude", "longitude",
    "country_enc", "field_type_EGS",
    "field_type_Liquid-Dominated", "field_type_Vapor-Dominated",
]
TARGET = "power_generated_MWe"

X = prod[FEATURES].values
y = prod[TARGET].values
print(f"      ✓ Features: {len(FEATURES)} | Samples: {len(X):,}")

# ── 3. Train / Val / Test Split (70 / 15 / 15) ────────────────────────────────
print("[3/5] Splitting data...")
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.1765, random_state=42)
print(f"      ✓ Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")

# ── 4. GridSearchCV — Random Forest Only ──────────────────────────────────────
print("[4/5] Running GridSearchCV for Random Forest...")
print("      (This may take 2–5 minutes depending on your machine)\n")

rf_pipe = Pipeline([
    ("rf", RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    "rf__n_estimators":     [100, 200],
    "rf__max_depth":        [10, 20, None],
    "rf__max_features":     ["sqrt", 0.5],
    "rf__min_samples_leaf": [1, 3],
}

gs = GridSearchCV(
    rf_pipe,
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    refit=True,
    verbose=1,
)
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
print(f"\n      ✓ Best params : {gs.best_params_}")

# ── Evaluate on Val & Test ────────────────────────────────────────────────────
yp_val  = best_model.predict(X_val)
yp_test = best_model.predict(X_test)

val_r2   = r2_score(y_val,  yp_val)
val_rmse = np.sqrt(mean_squared_error(y_val,  yp_val))
val_mae  = mean_absolute_error(y_val,  yp_val)

test_r2   = r2_score(y_test,  yp_test)
test_rmse = np.sqrt(mean_squared_error(y_test, yp_test))
test_mae  = mean_absolute_error(y_test, yp_test)

print(f"\n      Validation  → R²={val_r2:.4f}  RMSE={val_rmse:.4f}  MAE={val_mae:.4f}")
print(f"      Test        → R²={test_r2:.4f}  RMSE={test_rmse:.4f}  MAE={test_mae:.4f}")

# ── 5. Save Artifacts ─────────────────────────────────────────────────────────
print("\n[5/5] Saving artifacts...")
joblib.dump(best_model, "model.joblib",         compress=3)
joblib.dump(le,         "label_encoder.joblib", compress=3)
print("      ✓ model.joblib         saved")
print("      ✓ label_encoder.joblib saved")

# ── Training Report ───────────────────────────────────────────────────────────
report = f"""
GEOTHERMAL POWER PREDICTOR — TRAINING REPORT
=============================================
Model      : Random Forest Regressor
Algorithm  : sklearn RandomForestRegressor inside Pipeline
Best Params: {gs.best_params_}

Dataset    : geothermal_wells_worldwide.csv
Target     : {TARGET}
Features   : {FEATURES}

Split      : Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}

Metrics:
  Validation  R²={val_r2:.4f}  RMSE={val_rmse:.4f} MWe  MAE={val_mae:.4f} MWe
  Test        R²={test_r2:.4f}  RMSE={test_rmse:.4f} MWe  MAE={test_mae:.4f} MWe

Saved Artifacts:
  model.joblib          — full trained Pipeline (scaler + RF)
  label_encoder.joblib  — LabelEncoder for country feature
"""

with open("training_report.txt", "w") as f:
    f.write(report)
print("      ✓ training_report.txt  saved")

print("\n" + "=" * 60)
print("  ✅ Training complete! Launch the app with:")
print("     streamlit run app.py")
print("=" * 60)