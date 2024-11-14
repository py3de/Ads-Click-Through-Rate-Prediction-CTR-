import pandas as pd
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
from config import DATA_PATH  # assuming config.py contains paths and settings

def load_data():
    # Load data
    data = pd.read_csv(DATA_PATH['processed_data'])  # Adjust as per your config

    # Drop or convert non-numeric columns
    if 'campaign_name' in data.columns:
        data.drop(columns=['campaign_name'], inplace=True)
    if 'status' in data.columns:
        data['status'] = data['status'].astype('category').cat.codes
    if 'date' in data.columns:
        data.drop(columns=['date'], inplace=True)
    if 'ad_id' in data.columns:
        data.drop(columns=['ad_id'], inplace=True)  # Drop the ad_id column

    # Replace 'click' with the actual column name that you want to predict
    X = data.drop(columns=['clicks'])  # Drop features column, assuming 'clicks' is the target
    y = data['clicks']  # The target column

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_xgboost(X_train, y_train, X_test, y_test):
    xgb_model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        eval_metric="rmse"  # Root Mean Square Error, suitable for regression
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    return xgb_model


# Train using LightGBM
def train_lightgbm(X_train, y_train, X_test, y_test):
    # Initialize LGBMRegressor
    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        max_depth=6,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        boosting_type='gbdt',
        random_state=42,
        n_estimators=500  # Number of boosting iterations
    )

    # Train the model
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
    )

    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Save the trained model
    model_path = os.path.join(model_dir, 'lgb_model.joblib')
    joblib.dump(lgb_model, model_path)

    return lgb_model


# Main training function
def main():
    X_train, X_test, y_train, y_test = load_data()
    
    print("Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    print("XGBoost training completed.")
    
    print("Training LightGBM model...")
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
    print("LightGBM training completed.")
    
    # Evaluate both models using RMSE (Root Mean Squared Error)
    xgb_preds = xgb_model.predict(X_test)
    lgb_preds = lgb_model.predict(X_test)
    
    # Calculate RMSE for both models
    xgb_rmse = mean_squared_error(y_test, xgb_preds, squared=False)
    lgb_rmse = mean_squared_error(y_test, lgb_preds, squared=False)
    
    print("XGBoost RMSE:", xgb_rmse)
    print("LightGBM RMSE:", lgb_rmse)

if __name__ == "__main__":
    main()