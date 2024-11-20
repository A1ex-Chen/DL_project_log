import argparse
import os
import sys
import time
import traceback
from aibox.lib.task.classification.config import Config as ClassificationConfig
from aibox.lib.task import Task
import uuid
from multiprocessing import Event
import pkg_resources
from aibox.lib.logger import Logger
from aibox.lib.config import Config
from aibox.lib.db import DB
from aibox.lib.augmenter import Augmenter
from aibox.lib.task.classification.algorithm import Algorithm as ClassificationAlgorithm
from aibox.lib.task.detection.config import Config as DetectionConfig
from aibox.lib.task.detection.algorithm import Algorithm as DetectionAlgorithm
from aibox.lib.task.detection.backbone import Backbone
from aibox.lib.task.instance_segmentation.algorithm import Algorithm as InstanceSegmentationAlgorithm
from aibox.lib.task.instance_segmentation.config import Config as InstanceSegmentationConfig
import torch




if __name__ == '__main__':
    main()