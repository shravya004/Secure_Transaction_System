from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Core pipeline
from app.core.secure_pipeline import process_transaction

# Crypto + Blockchain
from app.crypto_engine.key_manager import KeyManager
from app.blockchain.blockchain import Blockchain

# ML Predict Router
from app.routers import predict


app = FastAPI(title="AI-Blockchain Secure Transaction System")

# Initialize blockchain
blockchain = Blockchain()

# Register ML Predict Router
app.include_router(predict.router, prefix="/predict", tags=["AI Prediction"])


# =========================
# Request Model
# =========================
class TransactionRequest(BaseModel):
    features: list[float]
    amount: float
    sender: str
    receiver: str


# =========================
# Root Endpoint
# =========================
@app.get("/")
def root():
    return {
        "message": "AI Blockchain Secure Transaction API is running"
    }


# =========================
# Transaction Processing
# =========================
@app.post("/transaction")
def handle_transaction(request: TransactionRequest):

    transaction_data = {
        "features": request.features,
        "amount": request.amount,
        "sender": request.sender,
        "receiver": request.receiver
    }

    # Generate ECDSA keys
    private_key, public_key = KeyManager.generate_keypair()

    # Process transaction through secure pipeline
    result = process_transaction(
        transaction_data,
        private_key,
        public_key
    )

    return {
        "status": "success",
        "message": result
    }


# =========================
# View Blockchain
# =========================
@app.get("/blockchain")
def get_blockchain():
    return {
        "chain_length": len(blockchain.chain),
        "is_valid": blockchain.is_chain_valid(),
        "chain": blockchain.chain
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)