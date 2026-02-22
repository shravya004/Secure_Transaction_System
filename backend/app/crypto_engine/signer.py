# Placeholder file for signer.py
from ecdsa import SigningKey, SECP256k1
import binascii

def sign_transaction(private_key_hex, message_hash):
    private_key = SigningKey.from_string(
        binascii.unhexlify(private_key_hex),
        curve=SECP256k1
    )
    signature = private_key.sign(message_hash.encode())
    return binascii.hexlify(signature).decode()
