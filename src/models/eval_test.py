import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_models():
    try:
        lgb_model = joblib.load('lgb_model.joblib')
        print("LightGBM model loaded successfully.")
    except FileNotFoundError:
        print("LightGBM model not found.")
        lgb_model = None

    try:
        xgb_model = joblib.load('xgb_model.joblib')
        print("XGBoost model loaded successfully.")
    except FileNotFoundError:
        print("XGBoost model not found.")
        xgb_model = None

    return lgb_model, xgb_model

def evaluate_model_with_residuals(model, X_test, y_test, model_name="Model"):
    if model is None:
        print(f"{model_name} is not available for evaluation.")
        return
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Evaluation Metrics
    mAE = mean_absolute_error(y_test, predictions)
    mSE = mean_squared_error(y_test, predictions)
    rmSE = mean_squared_error(y_test, predictions, squared=False)
    
    print(f"\nEvaluation for {model_name}:")
    print(f"  Mean Absolute Error (MAE): {mAE:.4f}")
    print(f"  Mean Squared Error (MSE): {mSE:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmSE:.4f}")
    print("-" * 50)
    
    # Residuals
    residuals = y_test - predictions
    
    
    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals, alpha=0.7, edgecolors='k')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
    plt.title(f'Residual Plot for {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

# Simulate test data (Replace with actual test data)
X_test = np.random.rand(20, 4)  # Replace with your actual test features
y_test = np.random.rand(20) * 1000  # Replace with your actual test labels

# Load models
lgb_model, xgb_model = load_models()

# Evaluate models and plot residuals
evaluate_model_with_residuals(lgb_model, X_test, y_test, model_name="LightGBM")
evaluate_model_with_residuals(xgb_model, X_test, y_test, model_name="XGBoost")