
import requests
import json
import pandas as pd
from datetime import datetime

# Configuration - Replace these with your actual credentials
ACCESS_TOKEN = 'Access Token'  # Your Access Token only valid for 24 hrs
AD_ACCOUNT_ID = 'ad_account_id'  # Your Ad Account ID
BASE_URL = 'https://api.pinterest.com/v5/ad_accounts'  # Base URL for Pinterest API
ENDPOINT = f'{BASE_URL}/{AD_ACCOUNT_ID}/ads'  # Full endpoint for fetching ads data

# Set up headers with Bearer token
headers = {
    'Authorization': f'Bearer {ACCESS_TOKEN}',
    'Content-Type': 'application/json'
}

# Function to fetch data from Pinterest API
def fetch_pinterest_data():
    response = requests.get(ENDPOINT, headers=headers)
    
    if response.status_code == 200:
        print("Data fetched successfully!")
        return response.json()  # Return the JSON response
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Function to process and clean data
def process_data(data):
    # Example processing: Extracting relevant fields and structuring data
    if data:
        ads = data.get('data', [])  # Assuming 'data' is the key that holds the ads
        cleaned_data = []
        
        for ad in ads:
            ad_info = {
                'ad_id': ad.get('id', ''),
                'campaign_name': ad.get('name', ''),
                'status': ad.get('status', ''),
                'impressions': ad.get('impressions', 0),
                'clicks': ad.get('clicks', 0),
                'ctr': ad.get('ctr', 0),
                'spend': ad.get('spend', 0.0),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Timestamp for when data was fetched
            }
            cleaned_data.append(ad_info)
        
        return pd.DataFrame(cleaned_data)  # Convert to DataFrame for easier analysis
    else:
        print("No data to process.")
        return None

# Main function to execute the steps
def main():
    print("Fetching Pinterest ad campaign data...")
    data = fetch_pinterest_data()
    
    if data:
        # Process the data into a structured format
        cleaned_df = process_data(data)
        
        if cleaned_df is not None:
            print(f"Processed Data:\n{cleaned_df.head()}")  # Preview of the cleaned data
            # Optionally, save the data to a CSV for later use
            cleaned_df.to_csv('pinterest_ads_data.csv', index=False)
            print("Data saved as 'pinterest_ads_data.csv'")
        else:
            print("No valid data to process.")
    else:
        print("Failed to fetch data.")

if __name__ == "__main__":
    main()




