"""Train the Random Forest used by app.py and save it as rf_model.joblib.

The notebook explores the data and tunes hyperparameters; this script
reproduces the final model in one step (about 5 seconds on a laptop CPU).
It trains on raw (unscaled) features because app.py passes raw slider values
to the model; tree splits do not depend on feature scaling, so accuracy is
the same as the notebook's scaled version.

Usage:  python train.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "california_housing.csv"
MODEL_PATH = "rf_model.joblib"


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Train on an 80/20 split, save the model, and return (model, metrics)."""
    df = pd.read_csv(data_path).drop_duplicates()
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Best parameters found by GridSearchCV in the notebook.
    model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, preds),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    joblib.dump(model, model_path, compress=3)
    return model, metrics


if __name__ == "__main__":
    _, m = train_model()
    print(f"Test R2:   {m['r2']:.4f}")
    print(f"Test RMSE: {m['rmse']:.4f}  (~${m['rmse'] * 100_000:,.0f})")
    print(f"Saved {MODEL_PATH}")
