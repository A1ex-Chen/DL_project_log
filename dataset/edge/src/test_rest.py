import pickle
import struct
import cv2
import numpy as np
import time
from datetime import timedelta, datetime

import socket
import _thread
import serial
import json
from config import config
import requests

ip="192.168.55.1"
rabbitmq="192.168.0.106" #ip server chạy rabbitmq


_thread.start_new_thread(ReadData, ("ReadData",))
# def get_data(self, api):
#         response = requests.get(f"{api}")
#         if response.status_code == 200:
#             print("sucessfully fetched the data")
#             self.formatted_print(response.json())
#         else:
#             print(f"Hello person, there's a {response.status_code} error with your request")