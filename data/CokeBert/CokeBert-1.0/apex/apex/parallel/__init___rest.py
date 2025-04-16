import torch

if hasattr(torch.distributed, 'ReduceOp'):
    ReduceOp = torch.distributed.ReduceOp
elif hasattr(torch.distributed, 'reduce_op'):
    ReduceOp = torch.distributed.reduce_op
else:
    ReduceOp = torch.distributed.deprecated.reduce_op

from .distributed import DistributedDataParallel, Reducer
# This is tricky because I'd like SyncBatchNorm to be exposed the same way
# for both the cuda-enabled and python-fallback versions, and I don't want
# to suppress the error information.
try:
    import syncbn
    from .optimized_sync_batchnorm import SyncBatchNorm
except ImportError as err:
    from .sync_batchnorm import SyncBatchNorm
    SyncBatchNorm.syncbn_import_error = err

