import os
import json
from bitcoinlib.wallets import Wallet, wallet_delete_if_exists
from bitcoinlib.mnemonic import Mnemonic
from bitcoinlib.keys import HDKey
from bitcoinlib.transactions import Transaction
from bitcoinlib.services.services import Service

class CryptoWallet:
    def __init__(self, wallet_name, network='testnet', seed_phrase=None):
        self.wallet_name = wallet_name
        self.network = network
        self.seed_phrase = seed_phrase
        self.wallet = None

        if wallet_delete_if_exists(self.wallet_name):
            print(f"Existing wallet '{self.wallet_name}' deleted.")

        if seed_phrase:
            self.create_wallet_from_seed(seed_phrase)
        else:
            self.create_new_wallet()

    def create_wallet_from_seed(self, seed_phrase):
        try:
            key = HDKey.from_passphrase(seed_phrase, network=self.network)
            self.wallet = Wallet.create(
                self.wallet_name, keys=key, network=self.network, witness_type='segwit'
            )
            print(f"Wallet '{self.wallet_name}' restored from seed phrase.")
        except Exception as e:
            raise ValueError(f"Failed to create wallet from seed phrase: {str(e)}")

    def create_new_wallet(self):
        try:
            self.seed_phrase = Mnemonic().generate()
            key = HDKey.from_passphrase(self.seed_phrase, network=self.network)
            self.wallet = Wallet.create(
                self.wallet_name, keys=key, network=self.network, witness_type='segwit'
            )
            print(f"New wallet '{self.wallet_name}' created.")
            print(f"IMPORTANT: Save this seed phrase securely: {self.seed_phrase}")
        except Exception as e:
            raise ValueError(f"Failed to create new wallet: {str(e)}")

    def get_balance(self):
        self.wallet.scan()
        return self.wallet.balance()

    def get_address(self):
        return self.wallet.get_key().address

    def send_transaction(self, to_address, amount, fee=None):
        try:
            t = self.wallet.send_to(to_address, amount, fee=fee)
            t.sign()
            t.send()
            return {"txid": t.txid, "fee": t.fee}
        except Exception as e:
            return {"error": str(e)}

    def get_transaction_history(self):
        self.wallet.scan()
        transactions = []
        for t in self.wallet.transactions():
            transactions.append({
                "txid": t.txid,
                "confirmations": t.confirmations,
                "date": t.date.isoformat() if t.date else None,
                "amount": t.amount,
                "fee": t.fee,
                "inputs": [{"address": i.address, "amount": i.value} for i in t.inputs],
                "outputs": [{"address": o.address, "amount": o.value} for o in t.outputs],
            })
        return transactions

    def import_private_key(self, private_key):
        try:
            key = HDKey(private_key, network=self.network)
            self.wallet.import_key(key)
            print(f"Private key imported successfully.")
        except Exception as e:
            print(f"Failed to import private key: {str(e)}")

    def export_private_key(self, address):
        try:
            key = self.wallet.get_key(address)
            return key.wif
        except Exception as e:
            return f"Failed to export private key: {str(e)}"

    def backup_wallet(self, filename):
        backup_data = {
            "seed_phrase": self.seed_phrase,
            "network": self.network,
            "addresses": [key.address for key in self.wallet.keys()],
        }
        with open(filename, 'w') as f:
            json.dump(backup_data, f)
        print(f"Wallet backed up to {filename}")

    def restore_wallet(self, filename):
        with open(filename, 'r') as f:
            backup_data = json.load(f)
        self.seed_phrase = backup_data["seed_phrase"]
        self.network = backup_data["network"]
        self.create_wallet_from_seed(self.seed_phrase)
        print(f"Wallet restored from {filename}")

    def get_network_fee(self):
        srv = Service(network=self.network)
        return srv.estimatefee()

    def __str__(self):
        return f"CryptoWallet(name='{self.wallet_name}', network='{self.network}')"

if __name__ == "__main__":
    # Example usage
    wallet = CryptoWallet("test_wallet", network="testnet")
    print(f"Wallet address: {wallet.get_address()}")
    print(f"Wallet balance: {wallet.get_balance()}")