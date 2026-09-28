from typing import Any
import traceback

import pandas as pd
from fastapi import FastAPI, HTTPException

from inference import predict_fraud

app = FastAPI(
    title="IEEE-CIS Fraud Detection API",
    description="API for predicting fraudulent transactions.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "IEEE-CIS Fraud Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(transaction: dict[str, Any]):
    try:
        df = pd.DataFrame([transaction])
        probability = float(predict_fraud(df)[0])

        return {
            "fraud_probability": probability
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

 
@app.post("/predict_batch")
def predict_batch(transactions: list[dict[str, Any]]):
    if not transactions:
        raise HTTPException(
            status_code=400,
            detail="Transaction list cannot be empty"
        )

    try:
        df = pd.DataFrame(transactions)

        probabilities = predict_fraud(df)

        return {
            "total_transactions": len(transactions),
            "predictions": [
                float(p) for p in probabilities
            ]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail=str(e)
        )   