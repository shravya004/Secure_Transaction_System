import hashlib
import json
import os
import time


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHAIN_FILE = os.path.join(DATA_DIR, "blockchain.json")


class Blockchain:

    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.chain = self.load_chain()

        if not self.chain:
            self.create_genesis_block()
            self.save_chain()

    def create_genesis_block(self):
        genesis_block = {
            "index": 0,
            "timestamp": time.time(),
            "data": "Genesis Block",
            "previous_hash": "0",
            "nonce": 0,
            "hash": ""
        }
        genesis_block["hash"] = self.calculate_hash(genesis_block)
        self.chain.append(genesis_block)

    def calculate_hash(self, block):
        block_string = json.dumps(
            {k: block[k] for k in block if k != "hash"},
            sort_keys=True
        ).encode()
        return hashlib.sha256(block_string).hexdigest()

    def proof_of_work(self, block, difficulty=3):
        block["nonce"] = 0
        computed_hash = self.calculate_hash(block)

        while not computed_hash.startswith("0" * difficulty):
            block["nonce"] += 1
            computed_hash = self.calculate_hash(block)

        return computed_hash

    def add_block(self, data):
        previous_block = self.chain[-1]

        new_block = {
            "index": len(self.chain),
            "timestamp": time.time(),
            "data": data,
            "previous_hash": previous_block["hash"],
            "nonce": 0,
            "hash": ""
        }

        new_block["hash"] = self.proof_of_work(new_block)

        self.chain.append(new_block)
        self.save_chain()

        print(f"Block mined: {new_block['hash']}")

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            if current["hash"] != self.calculate_hash(current):
                return False

            if current["previous_hash"] != previous["hash"]:
                return False

        return True

    def save_chain(self):
        with open(CHAIN_FILE, "w") as f:
            json.dump(self.chain, f, indent=4)

    def load_chain(self):
        if os.path.exists(CHAIN_FILE):
            with open(CHAIN_FILE, "r") as f:
                return json.load(f)
        return []