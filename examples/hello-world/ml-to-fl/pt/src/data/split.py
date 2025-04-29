import pandas as pd
import numpy as np
from sklearn.utils import shuffle

# Read the dataset
print("Reading the dataset...")
df = pd.read_csv('/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/creditcard.csv')

# Analyze the original dataset
total_rows = len(df)
fraud_rows = len(df[df['Class'] == 1])
non_fraud_rows = len(df[df['Class'] == 0])

print(f"Total rows: {total_rows}")
print(f"Fraud (Class 1): {fraud_rows} ({fraud_rows/total_rows*100:.2f}%)")
print(f"Non-fraud (Class 0): {non_fraud_rows} ({non_fraud_rows/total_rows*100:.2f}%)")

# Separate fraud and non-fraud instances
fraud_df = df[df['Class'] == 1]
non_fraud_df = df[df['Class'] == 0]

# Shuffle both dataframes
fraud_df = shuffle(fraud_df, random_state=42)
non_fraud_df = shuffle(non_fraud_df, random_state=42)

# New distribution: site1 45%, site2 45%, server 10%
site1_percentage = 0.45
site2_percentage = 0.45
server_percentage = 0.10

# Calculate fraud distribution
site1_fraud_count = int(fraud_rows * site1_percentage)
site2_fraud_count = int(fraud_rows * site2_percentage)
server_fraud_count = fraud_rows - site1_fraud_count - site2_fraud_count  # Remainder goes to server

# Calculate non-fraud distribution
non_fraud_for_site1 = int(non_fraud_rows * site1_percentage)
non_fraud_for_site2 = int(non_fraud_rows * site2_percentage)
non_fraud_for_server = non_fraud_rows - non_fraud_for_site1 - non_fraud_for_site2  # Remainder goes to server

print(f"\nDistribution plan:")
print(f"Site 1: {site1_fraud_count} fraud + {non_fraud_for_site1} non-fraud ({site1_percentage*100:.0f}% of data)")
print(f"Site 2: {site2_fraud_count} fraud + {non_fraud_for_site2} non-fraud ({site2_percentage*100:.0f}% of data)")
print(f"Server: {server_fraud_count} fraud + {non_fraud_for_server} non-fraud ({server_percentage*100:.0f}% of data)")

# Create the splits
site1_fraud = fraud_df.iloc[:site1_fraud_count]
site2_fraud = fraud_df.iloc[site1_fraud_count:site1_fraud_count+site2_fraud_count]
server_fraud = fraud_df.iloc[site1_fraud_count+site2_fraud_count:]

site1_non_fraud = non_fraud_df.iloc[:non_fraud_for_site1]
site2_non_fraud = non_fraud_df.iloc[non_fraud_for_site1:non_fraud_for_site1+non_fraud_for_site2]
server_non_fraud = non_fraud_df.iloc[non_fraud_for_site1+non_fraud_for_site2:]

# Combine fraud and non-fraud for each site
site1_df = pd.concat([site1_fraud, site1_non_fraud])
site2_df = pd.concat([site2_fraud, site2_non_fraud])
server_df = pd.concat([server_fraud, server_non_fraud])

# Shuffle the datasets
site1_df = shuffle(site1_df, random_state=42)
site2_df = shuffle(site2_df, random_state=42)
server_df = shuffle(server_df, random_state=42)

# Verify fraud percentages and dataset sizes
print("\nFinal dataset distribution:")
for name, dataset in [("Site 1", site1_df), ("Site 2", site2_df), ("Server", server_df)]:
    total = len(dataset)
    fraud = len(dataset[dataset['Class'] == 1])
    non_fraud = len(dataset[dataset['Class'] == 0])
    print(f"{name}: {total} entries, {fraud} fraud ({fraud/total*100:.2f}%), {non_fraud} non-fraud")
    print(f"Percentage of original data: {total/total_rows*100:.2f}%")

# Calculate how much of the original data is in each file
total_distributed = len(site1_df) + len(site2_df) + len(server_df)
print(f"\nTotal rows distributed: {total_distributed} out of {total_rows} ({total_distributed/total_rows*100:.2f}%)")

# Save to CSV files
site1_df.to_csv('/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/site1.csv', index=False)
site2_df.to_csv('/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/site2.csv', index=False)
server_df.to_csv('/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/server.csv', index=False)

combine_df = pd.concat([site1_df, site2_df])
combine_df.to_csv('/home/khoa/Khoa/outsource/na_thesis/examples/hello-world/ml-to-fl/pt/src/data/train_valid.csv', index=False)

print("\nFiles saved as site1.csv, site2.csv, and server.csv")