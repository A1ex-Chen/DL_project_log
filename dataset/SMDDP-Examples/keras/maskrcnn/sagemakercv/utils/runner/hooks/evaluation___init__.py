def __init__(self, eval_dataset, annotations_file, per_epoch=True,
    tensorboard=True, include_mask_head=True, verbose=False):
    self.eval_dataset = eval_dataset
    self.annotations_file = annotations_file
    self.per_epoch = per_epoch
    if is_sm_dist():
        from smdistributed.dataparallel.tensorflow import get_worker_comm
        self.comm = get_worker_comm()
    else:
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
    self.tensorboard = tensorboard
    self.verbose = verbose
    self.include_mask_head = include_mask_head
