from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
import numpy as np
import time
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import FraudDetectorNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection",
    version="1.0.0"
)


MODEL_PATH = "model/best_pytorch_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
model_loaded_at = None


def load_model():
    global model, model_loaded_at
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train.py first.")
    m = FraudDetectorNN(input_dim=30)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.eval()
    m.to(DEVICE)
    model_loaded_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    logger.info(f"Model loaded from {MODEL_PATH}")
    return m


@app.on_event("startup")
def startup_event():
    global model
    model = load_model()



class Transaction(BaseModel):
    Time:   float = Field(..., example=0.0)
    V1:     float = Field(..., example=-1.3598)
    V2:     float = Field(..., example=-0.0728)
    V3:     float = Field(..., example=2.5363)
    V4:     float = Field(..., example=1.3782)
    V5:     float = Field(..., example=-0.3383)
    V6:     float = Field(..., example=0.4624)
    V7:     float = Field(..., example=0.2396)
    V8:     float = Field(..., example=0.0987)
    V9:     float = Field(..., example=0.3638)
    V10:    float = Field(..., example=0.0908)
    V11:    float = Field(..., example=-0.5516)
    V12:    float = Field(..., example=-0.6178)
    V13:    float = Field(..., example=-0.9913)
    V14:    float = Field(..., example=-0.3112)
    V15:    float = Field(..., example=1.4682)
    V16:    float = Field(..., example=-0.4704)
    V17:    float = Field(..., example=0.2079)
    V18:    float = Field(..., example=0.0258)
    V19:    float = Field(..., example=0.4040)
    V20:    float = Field(..., example=0.2514)
    V21:    float = Field(..., example=-0.0183)
    V22:    float = Field(..., example=0.2778)
    V23:    float = Field(..., example=-0.1105)
    V24:    float = Field(..., example=0.0669)
    V25:    float = Field(..., example=0.1285)
    V26:    float = Field(..., example=-0.1891)
    V27:    float = Field(..., example=0.1336)
    V28:    float = Field(..., example=-0.0211)
    Amount: float = Field(..., example=149.62)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud:          bool
    risk_level:        str
    inference_time_ms: float
    model_version:     str


class BatchRequest(BaseModel):
    transactions: list[Transaction]



def transaction_to_tensor(t: Transaction) -> torch.Tensor:
    features = [
        t.Time, t.V1, t.V2, t.V3, t.V4, t.V5, t.V6, t.V7,
        t.V8, t.V9, t.V10, t.V11, t.V12, t.V13, t.V14, t.V15,
        t.V16, t.V17, t.V18, t.V19, t.V20, t.V21, t.V22, t.V23,
        t.V24, t.V25, t.V26, t.V27, t.V28, t.Amount
    ]
    return torch.FloatTensor(features).unsqueeze(0).to(DEVICE)


def get_risk_level(prob: float) -> str:
    "parametrable risk levels based on probability thresholds"
    if prob < 0.3:
        return "LOW"
    elif prob < 0.7:
        return "MEDIUM"
    return "HIGH"



@app.get("/model/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_loaded_at": model_loaded_at,
        "model_path": MODEL_PATH,
        "device": str(DEVICE)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """Predict fraud probability for a single transaction."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    with torch.no_grad():
        tensor = transaction_to_tensor(transaction)
        prob   = model(tensor).item()

    inference_ms = round((time.time() - start) * 1000, 3)

    return PredictionResponse(
        fraud_probability = round(prob, 4),
        is_fraud          = prob >= 0.5,
        risk_level        = get_risk_level(prob),
        inference_time_ms = inference_ms,
        model_version     = "1.0.0"
    )


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    """Predict fraud probability for a batch of transactions."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.transactions) > 1000:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000")

    start = time.time()

    features = []
    for t in request.transactions:
        tensor = transaction_to_tensor(t)
        features.append(tensor)

    batch = torch.cat(features, dim=0)

    with torch.no_grad():
        probs = model(batch).squeeze().cpu().numpy()

    if probs.ndim == 0:
        probs = np.array([probs.item()])

    inference_ms = round((time.time() - start) * 1000, 3)

    results = []
    for prob in probs:
        results.append({
            "fraud_probability": round(float(prob), 4),
            "is_fraud":          float(prob) >= 0.5,
            "risk_level":        get_risk_level(float(prob))
        })

    return {
        "predictions":        results,
        "total_transactions": len(results),
        "fraud_count":        sum(1 for r in results if r["is_fraud"]),
        "inference_time_ms":  inference_ms
    }
