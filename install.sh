#!/bin/bash

echo "Starting installation for Ads Click-Through Rate (CTR) Prediction Project"

# Step 1: Clone the repository
echo "Cloning the repository..."
git clone https://github.com/py3de/Ads-Click-Through-Rate-Prediction-CTR-.git
cd Ads-Click-Through-Rate-Prediction-CTR- || exit

# Step 2: Set up virtual environment
echo "Setting up a Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Step 3: Install dependencies
echo "Installing required Python packages..."
pip install -r requirements.txt

# Step 4: Set up environment variables
echo "Creating a .env file for environment variables..."
cat <<EOT >> .env
# Pinterest API Credentials
PINTEREST_ACCESS_TOKEN=your_access_token_here
PINTEREST_CLIENT_ID=your_client_id_here
PINTEREST_ACCOUNT_ID=your_ad_account_id_here

# Data Collection Configuration
DATA_FETCH_INTERVAL=30
API_REQUEST_LIMIT=500
RETRY_ATTEMPTS=3
REQUEST_TIMEOUT=10

# Model Training Parameters
TRAIN_TEST_SPLIT_RATIO=0.8
RANDOM_SEED=42
MODEL_SAVE_PATH=./models/ctr_model.pkl

# Logging and Debugging
LOGGING_LEVEL=INFO
LOG_FILE_PATH=./logs/project.log
EOT

echo "Installation completed successfully."
echo "Remember to replace placeholder values in the .env file with your actual API credentials."

# Final message
echo "You are ready to start the CTR prediction project!"
