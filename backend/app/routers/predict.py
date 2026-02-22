from fastapi import APIRouter
from pydantic import BaseModel
import uuid
from app.ml.inference import predict_transaction as model_predict

router = APIRouter()

class TransactionRequest(BaseModel):
    features: list[float]

@router.post("")
async def predict(req: TransactionRequest):
    risk_score = model_predict(req.features)

    status = "APPROVED" if risk_score < 0.5 else "REJECTED"

    return {
        "transaction_id": str(uuid.uuid4()),
        "risk_score": risk_score,
        "status": status,
        "message": "Real model prediction"
    }