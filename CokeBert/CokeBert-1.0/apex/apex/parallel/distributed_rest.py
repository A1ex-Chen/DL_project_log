import torch
import torch.distributed as dist
from torch.nn.modules import Module
from torch.autograd import Variable
from collections import OrderedDict
from itertools import chain
import copy
import importlib
from ..multi_tensor_apply import multi_tensor_applier

imported_flatten_impl = False




# apply_dist_call requires that tensors in 'bucket' are all the same type.



# flat_dist_call organizes 'tensors' by type.

            

        
class Reducer(object):
    """
    :class:`apex.parallel.Reducer` is a simple class that helps allreduce a module's parameters
    across processes.  :class:`Reducer` is intended to give the user additional control:
    Unlike :class:`DistributedDataParallel`, :class:`Reducer` will not automatically allreduce
    parameters during ``backward()``.
    Instead, :class:`Reducer` waits for the user to call ``<reducer_instance>.reduce()`` manually.
    This enables, for example, delaying the allreduce to be carried out every 
    several iterations instead of every single iteration.

    Like :class:`DistributedDataParallel`, :class:`Reducer` averages any tensors it allreduces 
    over the number of participating processes.

    :class:`Reducer` is designed to work with the upstream launch utility script 
    ``torch.distributed.launch`` with ``--nproc_per_node <= number of gpus per node``.
    When used with this launcher, :class:`Reducer` assumes 1:1 mapping of processes to GPUs.
    It also assumes that your script calls ``torch.cuda.set_device(args.rank)`` before creating the model.

    Args:
        module_or_grads_list: Either a network definition (module) being run in multi-gpu/distributed mode, or an iterable of gradients to be reduced.  If a module is passed in, the Reducer constructor will sync the parameters across processes (broadcasting from rank 0) to make sure they're all initialized with the same values.  If a list of gradients (that came from some module) is passed in, the user is responsible for manually syncing that module's parameters at the beginning of training.
    """
    
            
            
            
class DistributedDataParallel(Module):
    """
    :class:`apex.parallel.DistributedDataParallel` is a module wrapper that enables
    easy multiprocess distributed data parallel training, similar to ``torch.nn.parallel.DistributedDataParallel``.  Parameters are broadcast across participating processes on initialization, and gradients are
    allreduced and averaged over processes during ``backward()``.

    :class:`DistributedDataParallel` is optimized for use with NCCL.  It achieves high performance by 
    overlapping communication with computation during ``backward()`` and bucketing smaller gradient
    transfers to reduce the total number of transfers required.

    :class:`DistributedDataParallel` is designed to work with the upstream launch utility script 
    ``torch.distributed.launch`` with ``--nproc_per_node <= number of gpus per node``.
    When used with this launcher, :class:`DistributedDataParallel` assumes 1:1 mapping of processes to GPUs.
    It also assumes that your script calls ``torch.cuda.set_device(args.rank)`` before creating the model.

    https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed shows detailed usage.
    https://github.com/NVIDIA/apex/tree/master/examples/imagenet shows another example
    that combines :class:`DistributedDataParallel` with mixed precision training.

    Args:
        module: Network definition to be run in multi-gpu/distributed mode.
        message_size (int, default=1e7): Minimum number of elements in a communication bucket.
        delay_allreduce (bool, default=False):  Delay all communication to the end of the backward pass.  This disables overlapping communication with computation.
        allreduce_trigger_params (list, optional, default=None):  If supplied, should contain a list of parameters drawn from the model.  Allreduces will be kicked off whenever one of these parameters receives its gradient (as opposed to when a bucket of size message_size is full).  At the end of backward(), a cleanup allreduce to catch any remaining gradients will also be performed automatically.  If allreduce_trigger_params is supplied, the message_size argument will be ignored.
        allreduce_always_fp32 (bool, default=False):  Convert any FP16 gradients to FP32 before allreducing.  This can improve stability for widely scaled-out runs.
        gradient_average (bool, default=True):  Option to toggle whether or not DDP averages the allreduced gradients over processes.  For proper scaling, the default value of True is recommended.
        gradient_predivide_factor (float, default=1.0):  Allows perfoming the average of gradients over processes partially before and partially after the allreduce.  Before allreduce:  ``grads.mul_(1.0/gradient_predivide_factor)``.  After allreduce:  ``grads.mul_(gradient_predivide_factor/world size)``.  This can reduce the stress on the dynamic range of FP16 allreduces for widely scaled-out runs.

    .. warning::
        If ``gradient_average=False``, the pre-allreduce division (``grads.mul_(1.0/gradient_predivide_factor)``) will still be applied, but the post-allreduce gradient averaging (``grads.mul_(gradient_predivide_factor/world size)``) will be omitted.

    """







      
    # Broadcast rank 0's bucket structure across all processes, and have all processes 
    # regenerate their bucket structures to match. 
        
        

    






        


           

        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                        
                    grad_acc.register_hook(allreduce_hook)
                    self.grad_accs.append(grad_acc)

                wrapper(param)

    def allreduce_bucket(self, bucket):
        tensor = flatten(bucket)

        tensor_to_allreduce = tensor 

        if self.allreduce_always_fp32:
            tensor_to_allreduce = tensor.float() 

        if self.gradient_predivide_factor != 1.0:
            tensor_to_allreduce.mul_(1./self.gradient_predivide_factor)

        dist.all_reduce(tensor_to_allreduce)

        if self.gradient_average:
            if self.gradient_predivide_factor != self.world_size:
                tensor_to_allreduce.mul_(self.gradient_predivide_factor/self.world_size)

        if self.allreduce_always_fp32 and tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)
 
        return tensor
    

    def allreduce_maybe_retain(self, bucket, bucket_idx=-1):
        allreduced = self.allreduce_bucket(bucket)
        if self.retain_allreduce_buffers:
            if self.allreduce_buffers[bucket_idx] is not None:
                raise RuntimeError("The backward pass is attempting to replace an already-filled "
                                   "allreduce buffer.  This is almost certainly an error.")
            self.allreduce_buffers[bucket_idx] = allreduced
        else:
            if multi_tensor_applier.available:
                multi_tensor_applier(
                    self.multi_tensor_scale,
                    self._overflow_buf,
                    [unflatten(allreduced, bucket), bucket],
                    1.0)
            else:
                for buf, synced in zip(bucket, unflatten(allreduced, bucket)):
                    buf.copy_(synced)


    def allreduce_fallback(self):
        grads = [param.grad.data for param in self.module.parameters() if param.grad is not None]

        split_buckets = split_half_float_double(grads)

        # If retain_allreduce_buffers is True and delay_allreduce is False,
        # this will only be done during the first backward pass, ignored by the 
        # training script, and overwritten in the next forward pass.  So it's harmless. 
        if self.retain_allreduce_buffers:
            self.allreduce_buffers = [None for _ in range(len(split_buckets))]
    
        for i, bucket in enumerate(split_buckets):
            allreduced = self.allreduce_maybe_retain(bucket, i)


    def comm_ready_buckets(self, param):
        # Need to do this in every hook for compatibility with Ruberry's streaming backward PR.
        # self.reduction_stream.wait_stream(torch.cuda.current_stream())

        bucket_idx, bucket_loc = self.param_id_to_bucket[id(param)]

        if self.buckets[bucket_idx][bucket_loc] is not None:
            raise RuntimeError("The backward pass is attempting to replace an already-filled "
                               "bucket slot.  This is almost certainly an error.")

        self.buckets[bucket_idx][bucket_loc] = param.grad.data
        self.buckets_ready_size[bucket_idx] += 1

        if self.buckets_ready_size[bucket_idx] == self.bucket_sizes[bucket_idx]:
            if bucket_idx == self.next_bucket:
                torch.cuda.current_stream().record_event(self.reduction_event)
                self.reduction_stream.wait_event(self.reduction_event)
                with torch.cuda.stream(self.reduction_stream):
                    self.allreduce_maybe_retain(self.buckets[bucket_idx], bucket_idx)

                    self.next_bucket += 1

                    # Reversing upstream's logic here, because we constructed our buckets based on
                    # the order things were received during backward.
                    if len(self.ready_buckets_not_reduced) > 0:
                        sorted_todo = sorted(self.ready_buckets_not_reduced)
                        for i in sorted_todo:
                            # Nothing can be reduced now
                            if i > self.next_bucket:
                                break
                            elif i == self.next_bucket:
                                self.allreduce_maybe_retain(self.buckets[i], i)
                                self.ready_buckets_not_reduced.remove(i)
                                self.next_bucket += 1 
                            else:
                                raise ValueError("i should always be >= next_bucket")
            else:
                self.ready_buckets_not_reduced.add(bucket_idx)

        
    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
       
        if not self._disable_allreduce:
            if not self.delay_allreduce:
                param_list = [param for param in self.module.parameters() if param.requires_grad]

                # Conditions under which to refresh self.record
                # Forward has the authority to set needs_refresh to True, but only allreduce_params
                # in backward has the authority to set needs_refresh to False.
                # Parentheses are not necessary for correct order of operations, but make the intent clearer.
                if ((not self.active_params) or 
                    (len(param_list) != len(self.active_params)) or
                    any([param1 is not param2 for param1, param2 in zip(param_list, self.active_params)])):
                    self.needs_refresh = True

                if self.needs_refresh:
                    self.active_i_buckets = []
                    self.buckets = []
                    self.tmp_buckets = [[], [], []] # [running half, float, double buckets]
                    self.tmp_numels = [0, 0, 0]
                    self.bucket_sizes = []
                    self.param_id_to_active_i = {id(param) : i for i, param in enumerate(param_list)}  
                    self.param_id_to_bucket = {}
                else:
                    self.buckets = [[None for _ in range(self.bucket_sizes[i])] 
                                   for i in range(self.num_buckets)] 
                    self.buckets_ready_size = [0 for i in range(self.num_buckets)]
                    if(self.retain_allreduce_buffers):
                        self.allreduce_buffers = [None for _ in range(self.num_buckets)]
                    self.next_bucket = 0
                    self.ready_buckets_not_reduced = set()
            
                self.active_params = param_list

            self.callback_queued = False
        
        return result