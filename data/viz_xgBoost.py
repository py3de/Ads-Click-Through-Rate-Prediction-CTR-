import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Simulate Synthetic dataset for XGBoostRegressor model
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    "Impressions": np.random.randint(1000, 100000, size=n_samples),
    "Clicks": np.random.randint(10, 1000, size=n_samples),
    "Spending": np.random.uniform(100, 1000, size=n_samples),
    "Conversions": np.random.randint(1, 500, size=n_samples)
})

# Features and Target
X = data[["Impressions", "Clicks", "Spending"]]
y = data["Conversions"]


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = load("src/models/xgb_model.joblib")


def plot_learning_curve(model, X_train, y_train, X_val, y_val, title):
    train_errors = []
    val_errors = []
    
 
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    for train_size in train_sizes:
        subset_size = int(train_size * len(X_train))
        

        X_train_subset = X_train[:subset_size]
        y_train_subset = y_train[:subset_size]
        

        model.fit(
            X_train_subset, 
            y_train_subset,
            eval_metric="rmse",
            early_stopping_rounds=10,
            eval_set=[(X_val, y_val)],
            verbose=True
        )
        

        y_train_pred = model.predict(X_train_subset)
        y_val_pred = model.predict(X_val)
        

        train_rmse = np.sqrt(mean_squared_error(y_train_subset, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        
        # Store errors
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
    

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors, label="Training RMSE", marker="o")
    plt.plot(train_sizes, val_errors, label="Validation RMSE", marker="o")
    plt.title(f"Learning Curve - {title}")
    plt.xlabel("Training Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.show()


plot_learning_curve(xgb_model, X_train, y_train, X_val, y_val, "XGBoost")