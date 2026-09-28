
import pandas as pd
import requests

# Load test data
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

# Select five transactions
sample = test_df.head(5)

# Convert to JSON records, with missing values as null
records = sample.to_json(orient="records")
import json
payload = json.loads(records)

# Send one batch request
url = "http://127.0.0.1:8000/predict_batch"

try:
    response = requests.post(
        url,
        json=payload,
        timeout=300
    )

    response.raise_for_status()
    result = response.json()

    print("Batch prediction successful!")
    print("Total transactions:", result["total_transactions"])
    print("Predictions:", result["predictions"])

except requests.RequestException as e:
    print("Request failed:", e)
    if e.response is not None:
        print("Error details:", e.response.text)