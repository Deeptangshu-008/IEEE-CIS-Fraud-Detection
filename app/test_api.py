
import json
import urllib.request
import urllib.error

import pandas as pd

# Load the test dataset
test_transaction = pd.read_csv("data/test_transaction.txt")
test_identity = pd.read_csv("data/test_identity.csv")

# Normalize identity column names
test_identity.columns = test_identity.columns.str.replace(
    "-", "_", regex=False
)

# Merge transaction and identity data
test_df = test_transaction.merge(
    test_identity,
    on="TransactionID",
    how="left"
)

test_df.head(5).to_csv(
    "data/api_test_sample.csv",
    index=False
)


# Select one real transaction
row = test_df.iloc[0].to_dict()

# Replace missing values with None and convert NumPy values
payload = {}
for key, value in row.items():
    if pd.isna(value):
        payload[key] = None
    elif hasattr(value, "item"):
        payload[key] = value.item()
    else:
        payload[key] = value

# Send transaction to the API
url = "http://127.0.0.1:8000/predict"
data = json.dumps(payload).encode("utf-8")

request = urllib.request.Request(
    url,
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST"
)

try:
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))
        print("API response:", result)

except urllib.error.HTTPError as e:
    print("HTTP Status:", e.code)
    print("Error details:", e.read().decode("utf-8"))

except Exception as e:
    print("Request failed:", e)