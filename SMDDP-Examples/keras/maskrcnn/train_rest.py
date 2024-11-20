import sys
sys.path.append('..')
import os
import argparse
from configs import cfg
from sagemakercv.detection import build_detector
from sagemakercv.training import build_optimizer, build_scheduler, build_trainer
from sagemakercv.data import build_dataset
from sagemakercv.utils.dist_utils import get_dist_info, MPI_size, is_sm_dist
from sagemakercv.utils.runner import Runner, build_hooks
import tensorflow as tf

rank, local_rank, size, local_size = get_dist_info()
devices = tf.config.list_physical_devices('GPU')
for device in devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.set_visible_devices([devices[local_rank]], 'GPU')
logical_devices = tf.config.list_logical_devices('GPU')




if __name__=='__main__':
    args = parse()
    cfg.merge_from_file(args.config)
    assert cfg.INPUT.TRAIN_BATCH_SIZE%MPI_size()==0, f"Batch {cfg.INPUT.TRAIN_BATCH_SIZE} on {MPI_size()} GPUs"
    assert cfg.INPUT.EVAL_BATCH_SIZE%MPI_size()==0, f"Batch {cfg.INPUT.EVAL_BATCH_SIZE} on {MPI_size()} GPUs"
    main(cfg)