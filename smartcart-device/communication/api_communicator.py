import requests
import os

class api_communicator(object):
    def __init__(self, account):
        self.api_address = os.environ.get("REQUEST_ADDRESS")
        self.account = account

    def send_test_request(self):
        payload = {"Test": "Request"}
        requests.post(self.api_address, data=payload)