import torch
import torch.distributed as dist
from torch.nn import Parameter
from torch.nn import Module
from apex.parallel import DistributedDataParallel as DDP
import argparse
import os


parser = argparse.ArgumentParser(description='allreduce hook example')
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

if args.distributed:
    args.gpu = args.local_rank % torch.cuda.device_count()
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()

torch.set_printoptions(precision=10)
torch.manual_seed(args.local_rank)

class Model(Module):

model = Model()
# model = DDP(model, message_size=1, gradient_predivide_factor=8.0)
model = DDP(model, delay_allreduce=True)
# model = DDP(model, message_size=1, allreduce_trigger_params=[model.b])

x = torch.cuda.FloatTensor(4096*4096)

passed = True
torch.cuda.cudart().cudaProfilerStart()
for i in range(10):
    x.fill_(i + args.local_rank) # fill x with new values every iteration for sanity
    model.zero_grad()
    out = model(x)
    loss = out.sum()
    # torch.cuda.nvtx.range_push("backward")
    loss.backward()
    # torch.cuda.nvtx.range_pop()
    
    # torch.cuda.nvtx.range_push("synchronize() + info")
    # torch.cuda.synchronize()
    print("i = {}".format(i))
    if not info("model.a", model.module.a, 2.):  passed = False
    if not info("model.b", model.module.b, 1.):  passed = False
    # torch.cuda.nvtx.range_pop()
torch.cuda.cudart().cudaProfilerStop()

print("passed = ", passed)