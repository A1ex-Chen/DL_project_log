# python function.py [--graph-after-ddp] [--graph-before-ddp]
# python -m torch.distributed.launch --nproc_per_node=2 function.py [--graph-after-ddp] [--graph-before-ddp]

import torch
#from torch._six import container_abcs
import types
from itertools import chain
import argparse
import os

# questions:
# is a custom autograd function or graphing around a backward call better?
# how to allow double backward?
# lazily capture as part of live backward, or not?
# capture all the way down to AccumulateGrad functions, or not?
# If yes, need to deal with params used in graphs and non-graphed regions,
# and DDP bucket-slot-ready flags.  To help, user could supply a list of params
# known to be exclusive to the graphed region.

# Current limitation:  Assumes all args are Tensors.
# Arg tensors may or may not require grad.
# Any temporaries created in func_or_module must not be used
# outside func_or_module unless they are among func_or_module's
# explicit return values.



        @staticmethod

    if was_module:
        func_or_module.forward_eager = func_or_module.forward
        func_or_module.forward = types.MethodType(functionalized, func_or_module)
        return func_or_module
    else:
        return Graphed.apply


def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--graph-before-ddp", action="store_true")
    parser.add_argument("--graph-after-ddp", action="store_true")
    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.local_rank + 1)
    torch.cuda.manual_seed(args.local_rank + 1)

    print("{} graph_before_ddp {} graph_after_ddp {}\n".format(args.local_rank,
                                                               args.graph_before_ddp,
                                                               args.graph_after_ddp),
          flush=True)

    N, D_in, H, D_out = 640, 4096, 2048, 1024

    stream = torch.cuda.Stream()

    model_segment1 = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                                torch.nn.Dropout(p=0.2),
                                torch.nn.Dropout(p=0.4)).cuda()

    model_segment2 = torch.nn.Sequential(torch.nn.Linear(H, D_out),
                                torch.nn.Dropout(p=0.3),
                                torch.nn.Dropout(p=0.1)).cuda()

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(chain(model_segment1.parameters(),
                                      model_segment2.parameters()),
                                lr = 0.1)

    x = torch.randn(N, D_in, device='cuda')
    h = torch.randn(N, H, device='cuda')
    y = torch.randn(N, D_out, device='cuda')

    pure_eager = not (args.graph_before_ddp or args.graph_after_ddp)

    if args.graph_before_ddp or pure_eager:
        print("Calling graph() before ddp\n")
        model_segment1 = graph(model_segment1,
                               (x.clone(),),
                               stream,
                               warmup_only=pure_eager)

        model_segment2 = graph(model_segment2,
                               (h.clone().requires_grad_(),),
                               stream,
                               warmup_only=pure_eager)

    model = torch.nn.Sequential(model_segment1, model_segment2)
    if args.distributed:
        # Small bucket cap to stress DDP
        torch.cuda.nvtx.range_push("DDP")
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          bucket_cap_mb=1,
                                                          device_ids=[args.local_rank],
                                                          gradient_as_bucket_view=True)
        torch.cuda.nvtx.range_pop()

    if args.graph_after_ddp:
        if args.distributed:
            print("Calling graph() after ddp\n")
            model.module[0] = graph(model.module[0], (x.clone(),), stream)
        else:
            model[0] = graph(model_segment1, (x.clone(),), stream)

    for e in range(2):
        model.train()
        for i in range(10):
            torch.cuda.nvtx.range_push("{}".format(i))
            optimizer.zero_grad(set_to_none=True)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            torch.cuda.nvtx.range_push("backward")
            loss.backward()
            torch.cuda.nvtx.range_pop()

            # possibly needed if post-backward sync is commented out in pytorch
            # torch.cuda.synchronize()

            torch.cuda.nvtx.range_push("step")
            optimizer.step()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_pop()

        print("train: {} {} {} {}".format(args.local_rank,
                                          loss.item(),
                                          tuple(p.grad.sum().item() for p in model_segment1.parameters()),
                                          tuple(p.grad.sum().item() for p in model_segment2.parameters())),
              flush=True)

        # do eval end of epoch
        with torch.no_grad():
            model.eval()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
        print("eval: {}\n".format(loss))

if __name__ == "__main__":
    main()