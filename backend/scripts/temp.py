from app.crypto_engine.key_manager import KeyManager
from app.crypto_engine.hash_utils import hash_transaction
from app.crypto_engine.signer import sign_transaction
from app.crypto_engine.verifier import verify_signature
from app.policy_engine.trust_policy import select_crypto_policy

# Simulated trust score from AI
trust_score = 0.42

policy = select_crypto_policy(trust_score)
print("Selected Policy:", policy)

# Generate keys
private_key, public_key = KeyManager.generate_keypair()
private_hex = KeyManager.serialize_private_key(private_key)
public_hex = KeyManager.serialize_public_key(public_key)

# Create transaction
transaction = {
    "amount": 500,
    "sender": "user_1",
    "receiver": "user_2"
}

# Hash transaction
tx_hash = hash_transaction(transaction)

# Sign
signature = sign_transaction(private_hex, tx_hash)

# Verify
is_valid = verify_signature(public_hex, signature, tx_hash)

print("Signature Valid:", is_valid)
