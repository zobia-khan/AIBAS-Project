
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Step 1: Scrape Data
url = "https://github.com/zobia-khan/AIBAS_Project_data"
response = requests.get(url)

if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    # Extracting all table data
    tables = soup.find_all('table')  # Finding all tables
    
    # Check the type and length of tables
    print(f"Type of tables: {type(tables)}")  # Debugging
    print(f"Number of tables found: {len(tables)}")  # Debugging
    
    if isinstance(tables, list) and len(tables) >= 2:  # Ensure tables is a list and has at least two elements
        # Select the second table
        second_table = tables[1]
        
        # Initialize data list
        data = []
        
        # Extract headers
        headers = [th.text.strip() for th in second_table.find_all('th')]
        
        # Extract each row's data
        for row in second_table.find_all('tr'):
            cells = row.find_all('td')
            if cells:
                data.append([cell.text.strip() for cell in cells])
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=headers)
        # print(df)
        print(df.info())
        print(df.head())
    else:
        raise Exception("Less than two tables found on the webpage or tables is not a list")
else:
    raise Exception("Failed to fetch webpage content")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
import os

# Ensure the directory exists
output_dir = '../../data/customer_churn_dataset'
os.makedirs(output_dir, exist_ok=True)

# Assume df is your DataFrame loaded from an earlier step
# Example placeholder: df = pd.read_csv('your_data.csv')

# Drop unnecessary columns
df.drop(['Surname', 'RowNumber', 'CustomerId'], axis=1, inplace=True)

# Check and handle missing or invalid values in 'Card Type'
df['Card Type'].replace('', np.nan, inplace=True)
df.dropna(subset=['Card Type'], inplace=True)
valid_categories = ["SILVER", "GOLD", "PLATINUM", "DIAMOND"]
df = df[df['Card Type'].isin(valid_categories)]

# Encode 'Card Type' using Ordinal Encoding
df['Card Type'] = OrdinalEncoder(
    categories=[valid_categories],
    dtype=int
).fit_transform(df[['Card Type']])

# One-Hot Encode 'Geography'
ohe = OneHotEncoder(sparse_output=False, dtype=int)
encoded = ohe.fit_transform(df[['Geography']])
encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())
df = pd.concat([df, encoded], axis=1)
df.drop(['Geography'], axis=1, inplace=True)

# Replace 'Gender' values with numeric equivalents
df.replace({'Female': 0, 'Male': 1}, inplace=True)

# Step 1: Convert relevant columns to numeric
numeric_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                   'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Satisfaction Score', 'Point Earned']

# Convert selected columns to numeric, coerce errors for non-numeric entries
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Step 2: Algorithmic Outlier Removal using numpy on selected columns
mean = np.mean(df[numeric_columns], axis=0)
std_dev = np.std(df[numeric_columns], axis=0)
threshold = 3  # Outlier threshold
outliers = np.abs((df[numeric_columns] - mean) / std_dev) > threshold
df_cleaned = df[~outliers.any(axis=1)]  # Keep rows where no column is an outlier

# Step 3: Normalization on selected columns
scaler = StandardScaler()
df_cleaned[numeric_columns] = scaler.fit_transform(df_cleaned[numeric_columns])

# Step 4: Store Cleaned Data
df_cleaned.to_csv(os.path.join(output_dir, 'joint_data_collection.csv'), index=False)

# Step 5: Split the Data
train, test = train_test_split(df_cleaned, test_size=0.2, random_state=42)

# Save the splits
train.to_csv(os.path.join(output_dir, 'training_data.csv'), index=False)
test.to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

# Step 6: Create Activation Data
# Select one random entry from the test set
activation_entry = test.sample(n=1)
activation_entry.to_csv(os.path.join(output_dir, 'activation_data.csv'), index=False)
