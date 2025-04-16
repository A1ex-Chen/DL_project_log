import base64
import numpy as np
import sys
import os
import json
import time
from config import config
import cv2
import pika

from utils.general import LOGGER

# Create connection
print('Creating connection...')
url = os.environ.get("CLOUDAMQP_URL", f"amqp://admin:admin@{config.server_ip}:5672?heartbeat=900")
params = pika.URLParameters(url)
#params.socket_timeout = 5
connection = pika.BlockingConnection(params)
channel = connection.channel()
channel.queue_declare(queue="q-3")
print('Connection established')


