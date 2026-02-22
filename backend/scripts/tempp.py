from app.blockchain.blockchain import Blockchain


def run_blockchain_test():
    blockchain = Blockchain()

    blockchain.add_block({"amount": 100, "sender": "Alice", "receiver": "Bob"})
    blockchain.add_block({"amount": 50, "sender": "Bob", "receiver": "Charlie"})

    print("Is blockchain valid?", blockchain.is_chain_valid())

    for block in blockchain.chain:
        print("\nBlock Index:", block.index)
        print("Hash:", block.hash)
        print("Previous Hash:", block.previous_hash)


if __name__ == "__main__":
    run_blockchain_test()
