# Placeholder file for key_manager.py
from ecdsa import SigningKey, SECP256k1

class KeyManager:

    @staticmethod
    def generate_keypair():
        private_key = SigningKey.generate(curve=SECP256k1)
        public_key = private_key.get_verifying_key()
        return private_key, public_key

    @staticmethod
    def serialize_public_key(public_key):
        return public_key.to_string().hex()

    @staticmethod
    def serialize_private_key(private_key):
        return private_key.to_string().hex()
