

# API Configuration
ACCESS_TOKEN = 'your_access_token_here'  # Pinterest API access token
AD_ACCOUNT_ID = 'your_ad_account_id_here'  # Pinterest ad account ID
BASE_URL = 'https://api.pinterest.com/v5/ad_accounts'  # Pinterest API base URL

# Data Settings
DATA_PATH={
'fetched_data':'pinterest_ads_data.csv',
'processed_data':'processed_data.csv',
'evaluation_data': 'evaluation_data.csv', #optional
}

# Model Parameters
MODEL_SAVE_PATH = 'models/ctr_prediction_model.pkl'  # Where to save the trained model
TEST_SIZE = 0.2  # Test dataset ratio
RANDOM_STATE = 42  # Random state for data shuffling and splitting
BATCH_SIZE = 64  # Batch size for training
EPOCHS = 100  # Number of epochs for model training


LOG_FILE_PATH = 'logs/ctr_model_logs.txt'  # Path to save log file


CLEANING_THRESHOLD = 0.1  # Threshold for cleaning missing values or outliers

# Feature Engineering Parameters
FEATURES = ['impressions', 'clicks', 'ctr', 'spend']  # Features for model training
TARGET = 'ctr'  # Target variable

# Model Hyperparameters
LEARNING_RATE = 0.001  # Learning rate for training
LAYER_SIZES = [128, 64, 32]  # Neural network layer sizes if using deep learning

# API Endpoints
ADS_ENDPOINT = f'{BASE_URL}/{AD_ACCOUNT_ID}/ads'  # Full endpoint for fetching ads data

# Experiment Settings
USE_SYNTHETIC_DATA = True  # Use synthetic data if actual data access is limited
DEBUG_MODE = False
