import hashlib

from app.crypto_engine.signer import sign_transaction
from app.crypto_engine.verifier import verify_signature
from app.policy_engine.trust_policy import select_policy
from app.ai_engine.predict import predict_risk
from app.blockchain.blockchain import Blockchain


# Initialize blockchain instance
blockchain = Blockchain()


def process_transaction(transaction_data, private_key, public_key):
    """
    End-to-end secure transaction pipeline:
    1. Hash transaction
    2. Sign using ECDSA
    3. Verify signature
    4. AI risk scoring
    5. Policy decision
    6. Append to blockchain if approved
    """

    # 🔹 Step 1: Hash transaction data
    message_hash = hashlib.sha256(
        str(transaction_data).encode()
    ).hexdigest()

    # 🔹 Step 2: Convert keys to hex
    private_key_hex = private_key.to_string().hex()
    public_key_hex = public_key.to_string().hex()

    # 🔹 Step 3: Sign transaction
    signature = sign_transaction(private_key_hex, message_hash)

    # 🔹 Step 4: Verify signature
    is_valid = verify_signature(public_key_hex, message_hash, signature)

    if not is_valid:
        return "❌ Signature Invalid — Transaction Rejected"

    # 🔹 Step 5: AI Risk Score
    risk_score = predict_risk(transaction_data["features"])

    # 🔹 Step 6: Policy Decision
    policy = select_policy(risk_score)

    if policy == "HIGH_RISK":
        return f"🚨 Transaction Rejected by AI — Risk Score: {risk_score}"

    # 🔹 Step 7: Add enriched transaction to blockchain
    enriched_transaction = {
        "transaction": transaction_data,
        "risk_score": risk_score,
        "signature": signature
    }

    blockchain.add_block(enriched_transaction)

    return f"✅ Transaction Approved — Risk Score: {risk_score}"
