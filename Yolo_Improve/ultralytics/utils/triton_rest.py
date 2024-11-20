# Ultralytics YOLO 🚀, AGPL-3.0 license

from typing import List
from urllib.parse import urlsplit

import numpy as np


class TritonRemoteModel:
    """
    Client for interacting with a remote Triton Inference Server model.

    Attributes:
        endpoint (str): The name of the model on the Triton server.
        url (str): The URL of the Triton server.
        triton_client: The Triton client (either HTTP or gRPC).
        InferInput: The input class for the Triton client.
        InferRequestedOutput: The output request class for the Triton client.
        input_formats (List[str]): The data types of the model inputs.
        np_input_formats (List[type]): The numpy data types of the model inputs.
        input_names (List[str]): The names of the model inputs.
        output_names (List[str]): The names of the model outputs.
    """

