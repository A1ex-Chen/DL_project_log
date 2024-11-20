import sys
import os
import json
sys.path.append('..')
import argparse
from configs import cfg
from sagemakercv.detection import build_detector
from sagemakercv.data import build_dataset
from sagemakercv.training import build_optimizer
from sagemakercv.utils.dist_utils import get_dist_info, MPI_size
from sagemakercv.data.coco import evaluation
import tensorflow as tf
import tensorflow_io as tfio

import smdistributed.dataparallel.tensorflow.keras as dist

dist.init()

rank, local_rank, size, local_size = get_dist_info()
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.set_visible_devices([devices[local_rank]], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": cfg.SOLVER.FP16})
tf.config.optimizer.set_jit(cfg.SOLVER.XLA)

sm_framework_params = os.environ.get('SM_FRAMEWORK_PARAMS', None)
if sm_framework_params is not None:
    sm_framework_params_dict = json.loads(sm_framework_params)
    instance_type = sm_framework_params_dict.get('sagemaker_instance_type', None)

# load backbone weights

# main training entry point

# distributed evaluation


if __name__=='__main__':
    args = parse()
    cfg.merge_from_file(args.config)
    assert cfg.INPUT.TRAIN_BATCH_SIZE%MPI_size()==0, f"Batch {cfg.INPUT.TRAIN_BATCH_SIZE} on {MPI_size()} GPUs"
    assert cfg.INPUT.EVAL_BATCH_SIZE%MPI_size()==0, f"Batch {cfg.INPUT.EVAL_BATCH_SIZE} on {MPI_size()} GPUs"
    main(cfg)