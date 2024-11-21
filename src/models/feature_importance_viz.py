import matplotlib.pyplot as plt
import numpy as np
import joblib

def plot_feature_importance(model, feature_names, model_name="Model"):
    """
    Plot the feature importance for a model.
    """
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]  # Sort by importance (descending)
    importance = importance[sorted_idx]
    feature_names = np.array(feature_names)[sorted_idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance, color='skyblue', edgecolor='k')
    plt.title(f'Feature Importance for {model_name}', fontsize=14)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.gca().invert_yaxis()  # Invert y-axis for descending order
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


lgb_model = joblib.load('lgb_model.joblib')
try:
    xgb_model = joblib.load('xgb_model.joblib')
except FileNotFoundError:
    print("XGBoost model not found.")
    xgb_model = None


feature_names = ['ad_id', 'campaign_name', 'status', 'impressions', 'clicks', 'ctr', 'spend', 'date']


if lgb_model:
    print("LightGBM Feature Importance:")
    plot_feature_importance(lgb_model, feature_names, model_name="LightGBM")

if xgb_model:
    print("XGBoost Feature Importance:")
    plot_feature_importance(xgb_model, feature_names, model_name="XGBoost")
