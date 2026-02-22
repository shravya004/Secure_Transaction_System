from app.crypto_engine.key_manager import KeyManager
from app.core.secure_pipeline import process_transaction


def run_test():

    private_key, public_key = KeyManager.generate_keypair()

    transaction = {
        "amount": 200,
        "sender": "Alice",
        "receiver": "Bob",
        "features": [0.3, 0.6, 0.1, 0.9]  # must match model input size
    }

    result = process_transaction(transaction, private_key, public_key)

    print(result)


if __name__ == "__main__":
    run_test()
