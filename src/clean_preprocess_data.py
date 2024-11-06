import pandas as pd

# Load the synthetic data (replace with the actual CSV file path)
df = pd.read_csv('Pinterest_ads_data.csv')

# Step 1: Check for missing values
print("Missing Values:\n", df.isnull().sum())  # Check for missing values in each column

# Step 2: Handle missing values (if any)
# Example: Filling missing values with appropriate strategies (mean, median, mode, or a constant value)
df['impressions'].fillna(df['impressions'].mean(), inplace=True)
df['clicks'].fillna(df['clicks'].median(), inplace=True)
df['ctr'].fillna(df['ctr'].mean(), inplace=True)
df['spend'].fillna(df['spend'].mean(), inplace=True)
df['date'].fillna(df['date'].mode()[0], inplace=True)

# Step 3: Check for duplicate rows
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Remove duplicates if any
df.drop_duplicates(inplace=True)

# Step 4: Convert columns to appropriate data types
df['ad_id'] = df['ad_id'].astype(str)  # Ensure 'ad_id' is a string
df['campaign_name'] = df['campaign_name'].astype(str)  # Ensure 'campaign_name' is a string
df['status'] = df['status'].astype(str)  # Ensure 'status' is a string
df['impressions'] = df['impressions'].astype(int)  # Ensure 'impressions' is an integer
df['clicks'] = df['clicks'].astype(int)  # Ensure 'clicks' is an integer
df['ctr'] = df['ctr'].astype(float)  # Ensure 'ctr' is a float
df['spend'] = df['spend'].astype(float)  # Ensure 'spend' is a float
df['date'] = pd.to_datetime(df['date'])  # Convert 'date' to datetime format

# Step 5: Handle outliers (example: using IQR method to filter out extreme values)
Q1 = df['ctr'].quantile(0.25)
Q3 = df['ctr'].quantile(0.75)
IQR = Q3 - Q1

# Define acceptable range (remove values that fall outside of this range)
df = df[(df['ctr'] >= (Q1 - 1.5 * IQR)) & (df['ctr'] <= (Q3 + 1.5 * IQR))]

# Step 6: Preview the cleaned data
print("\nCleaned Data Preview:\n", df.head())

# Step 7: Save the cleaned data
df.to_csv('cleaned_pinterest_ads_data.csv', index=False)
print("\nCleaned data saved as 'cleaned_pinterest_ads_data.csv'.")

