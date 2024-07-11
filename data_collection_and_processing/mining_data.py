import requests
import pandas as pd
from web3 import Web3


class EthereumTransactionAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        self.transactions = pd.read_csv(self.data_path)
        self.transactions_original = self.transactions.copy()

    def process_addresses(self):
        self.load_data()


