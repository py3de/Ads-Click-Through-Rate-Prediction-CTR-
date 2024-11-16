
import numpy as np
import joblib
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

def evaluate_model(model, X_test, y_test, model_name="Model"):
    if model is None:
        print(f"{model_name} is not available for evaluation.")
        return
    
    
    predictions = model.predict(X_test) # Predictions
    
    
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    print(f"\nEvaluation for {model_name}:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 50)



X_test = np.random.rand(20, 4)  # Replace with your actual -
y_test = np.random.rand(20) * 1000  #  test features

# Load models
lgb_model, xgb_model = load_models()

# Perform evaluation
evaluate_model(lgb_model, X_test, y_test, model_name="LightGBM")
evaluate_model(xgb_model, X_test, y_test, model_name="XGBoost")