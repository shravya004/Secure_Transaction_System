from ecdsa import VerifyingKey, SECP256k1
import binascii


def verify_signature(public_key_hex, message_hash, signature_hex):
    try:
        public_key = VerifyingKey.from_string(
            binascii.unhexlify(public_key_hex),
            curve=SECP256k1
        )

        signature = binascii.unhexlify(signature_hex)

        return public_key.verify(signature, message_hash.encode())

    except:
        return False
