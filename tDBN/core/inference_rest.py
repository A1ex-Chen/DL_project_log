import abc
import contextlib

import numpy as np
from google.protobuf import text_format

from tDBN.data.preprocess import merge_tDBN_batch, prep_pointcloud
from tDBN.protos import pipeline_pb2


class InferenceContext:



    @abc.abstractclassmethod


    @abc.abstractclassmethod


    @abc.abstractclassmethod


    @abc.abstractclassmethod

    @contextlib.contextmanager