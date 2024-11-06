import pandas as pd
import random
from datetime import datetime, timedelta

# Generate synthetic data for Pinterest ads
def generate_synthetic_data(num_records=100):
    campaigns = ["Campaign A", "Campaign B", "Campaign C", "Campaign D", "Campaign E"]
    statuses = ["Active", "Paused", "Completed"]
    
    # List to hold synthetic ad campaign data
    ads_data = []
    
    for _ in range(num_records):
        ad_info = {
            'ad_id': f'AD{random.randint(100000, 999999)}',
            'campaign_name': random.choice(campaigns),
            'status': random.choice(statuses),
            'impressions': random.randint(1000, 100000),
            'clicks': random.randint(100, 5000),
            'ctr': round(random.uniform(0.01, 0.10), 4),  # CTR between 1% and 10%
            'spend': round(random.uniform(100.0, 5000.0), 2),  # Spend between $100 and $5000
            'date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d %H:%M:%S')  # Random date in the last 30 days
        }
        ads_data.append(ad_info)
    
    # Convert to DataFrame for easy manipulation
    return pd.DataFrame(ads_data)

# Generate synthetic data
df_synthetic = generate_synthetic_data(num_records=100)  # Up to 100 records

# Save to CSV
df_synthetic.to_csv('Pinterest_ads_data.csv', index=False)

print("Synthetic Pinterest ads data has been saved as 'Pinterest_ads_data.csv'")
