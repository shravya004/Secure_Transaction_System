# Placeholder file for hash_utils.py
import hashlib
import json

def hash_transaction(transaction_data):
    tx_string = json.dumps(transaction_data, sort_keys=True)
    return hashlib.sha256(tx_string.encode()).hexdigest()
