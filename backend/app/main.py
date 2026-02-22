from fastapi import FastAPI
from pydantic import BaseModel
from app.core.secure_pipeline import process_transaction
from app.crypto_engine.key_manager import KeyManager
from app.blockchain.blockchain import Blockchain


app = FastAPI(title="AI-Blockchain Secure Transaction System")

blockchain = Blockchain()


class TransactionRequest(BaseModel):
    features: list
    amount: float
    sender: str
    receiver: str


@app.get("/")
def root():
    return {"message": "AI Blockchain Secure Transaction API is running"}


@app.post("/transaction")
def handle_transaction(request: TransactionRequest):

    transaction_data = {
        "features": request.features,
        "amount": request.amount,
        "sender": request.sender,
        "receiver": request.receiver
    }

    private_key, public_key = KeyManager.generate_keypair()

    result = process_transaction(transaction_data, private_key, public_key)

    return {
        "status": "success",
        "message": result
    }


@app.get("/blockchain")
def get_blockchain():
    return {
        "chain_length": len(blockchain.chain),
        "is_valid": blockchain.is_chain_valid(),
        "chain": blockchain.chain
    }