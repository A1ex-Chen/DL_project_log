import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from ..amp._amp_state import _amp_state, maybe_print
from ..amp.scaler import LossScaler
from ..multi_tensor_apply import multi_tensor_applier
from .fp16util import model_grads_to_master_grads, master_params_to_model_params, clip_grad_norm

# TODO:  Update overflow check + downscale to use Carl's fused kernel.
class FP16_Optimizer(object):
    """
    :class:`FP16_Optimizer` is designed to wrap an existing PyTorch optimizer, 
    and manage static or dynamic loss scaling and master weights in a manner transparent to the user.
    For standard use, only two lines must be changed:  creating the :class:`FP16_Optimizer` instance,
    and changing the call to ``backward``.

    Example::

        model = torch.nn.Linear(D_in, D_out).cuda().half()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        # Name the FP16_Optimizer instance to replace the existing optimizer
        # (recommended but not required):
        optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
        ...
        # loss.backward() becomes:
        optimizer.backward(loss)
        ...

    Example with dynamic loss scaling::

        ...
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
                                   # optional arg to control dynamic loss scaling behavior
                                   # dynamic_loss_args={'scale_window' : 500})
                                   # Usually, dynamic_loss_args is not necessary. 

    Args:
        init_optimizer (torch.optim.optimizer):  Existing optimizer created with the parameters to optimize.  Internally, :class:`FP16_Optimizer` replaces the passed optimizer's fp16 parameters, if any, with fp32 master parameters copied from the original ones.  :class:`FP16_Optimizer` also stores references to the original fp16 parameters, and updates these fp16 parameters from the master fp32 copy at the end of each :attr:`step`.  
        static_loss_scale (float, optional, default=1.0):  Loss scale used internally to scale gradients computed by the model.  Any fp16 gradients will be copied to fp32, then downscaled before being applied to the fp32 master params, so ``static_loss_scale`` should not affect learning rate.
        dynamic_loss_scale (bool, optional, default=False):  Use dynamic loss scaling.  If True, this will override any ``static_loss_scale`` option.
        dynamic_loss_args (dict, optional, default=None):  Dict of kwargs that will be forwarded to the internal :class:`LossScaler` instance's constructor.  Keys of this dict must match kwargs accepted by :class:`LossScaler`'s constructor.  If ``dynamic_loss_args`` is unspecified, :class:`LossScaler`'s defaults will be used.
        verbose (bool, optional, default=True):  By default, FP16_Optimizer's constructor prints out the parameters and parameter groups it is ingesting, as a sanity check.  If this becomes annoying (e.g. for large models), it can be disabled by passing ``verbose=False``.  ``verbose=False`` will not disable printing when the loss scale is readjusted during dynamic loss scaling.

    ``init_optimizer`` is expected to have been constructed in the ordinary way.  
    It is recommended (although not required) that the newly constructed :class:`FP16_Optimizer` instance be 
    named to replace ``init_optimizer``, for two reasons:  
    First, it means that references to the same name
    later in the file will not have to change.  
    Second, :class:`FP16_Optimizer` reserves the right (as an implementation detail) to 
    modify ``init_optimizer``.  If you do choose a unique name for the new
    :class:`FP16_Optimizer` instance, you should only work with this new instance,
    because the preexisting optimizer might no longer behave as expected.

    ``init_optimizer`` may be any Pytorch optimizer. 
    It may contain a mixture of fp16 and fp32 parameters organized into any number of 
    ``param_groups`` with different hyperparameters.  The :class:`FP16_Optimizer` constructor will 
    ingest these ``param_groups`` and remember them. 

    Calls to ::

        loss.backward() 

    must be replaced with ::

        optimizer.backward(loss)  

    because :class:`FP16_Optimizer` requires ownership of the backward pass to implement 
    loss scaling and copies to master gradients.

    .. note::
        Loss scaling, either static or dynamic, is orthogonal to learning rate, because gradients
        are downscaled before being applied.  This means that adjusting the loss scale, or using
        dynamic loss scaling, should not require retuning the learning rate or any other 
        hyperparameters.


    **Advanced options**

    **Closures**:  :class:`FP16_Optimizer` can wrap a Pytorch optimizer that receives a closure.
    See docstring for :attr:`step`.

    **Gradient clipping**:  Use :attr:`clip_master_grads`.
    
    **Multiple losses**:  If your model accumulates gradients from multiple losses,
    this can be made more efficient by supplying ``update_master_grads=False``
    to :attr:`backward`.  See docstring for :attr:`backward`.

    **Manually adjusting loss scale**:  The current loss scale can be retrieved or set via ::

        print(optimizer.loss_scale)
        optimizer.loss_scale = new_loss_scale

    For static loss scaling, manually adjusting the loss scale over time is a reasonable
    thing to do.  During later epochs, gradients may become smaller, and a 
    higher loss scale may be required, analogous to scheduling the learning rate.  Dynamic loss
    scaling is more subtle (see :class:`DynamicLossScaler`) and in this case, manually adjusting 
    the loss scale is not recommended.

    **Multi_GPU training**:  If the wrapped ``init_optimizer`` was created from a model wrapped in
    Pytorch DistributedDataParallel or Apex DistributedDataParallel, :class:`FP16_Optimizer` 
    should still work as intended.
    """


    # Having self.maybe_print distinct from _amp_state.maybe_print is another artifact
    # of having to support FP16_Optimizer separately, for the time being.
            



    # Should not be used anymore.
    # def _check_overflow(self):
    #     params = []
    #     for group in self.fp16_groups:
    #         for param in group:
    #             params.append(param)
    #     for group in self.fp32_from_fp32_groups:
    #         for param in group:
    #             params.append(param)
    #     self.overflow = self.loss_scaler.has_overflow(params)

    # def _update_scale(self, has_overflow=False):
    #     self.loss_scaler.update_scale(has_overflow)


    # To consider:  Integrate distributed with this wrapper by registering a hook on each variable
    # that does the overflow check, gradient copy + downscale, and fp32 allreduce in a different stream.
    # def _model_grads_to_master_grads(self):
    #     for fp16_group, fp32_from_fp16_group in zip(self.fp16_groups, self.fp32_from_fp16_groups):
    #         model_grads_to_master_grads(fp16_group, fp32_from_fp16_group)

    # def _downscale_master(self):
    #     if self.loss_scale != 1.0:
    #         for group in self.optimizer.param_groups:
    #             for param in group['params']:
    #                 if param.grad is not None:
    #                     param.grad.data.mul_(1./self.loss_scale)







        # torch.cuda.nvtx.range_pop()




    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"


    loss_scale = property(_get_loss_scale, _set_loss_scale)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"


    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)


        retval = self.optimizer.step(wrapped_closure)

        self.first_closure_call_this_step = True

        return retval

    def backward(self, loss, update_master_grads=True, retain_graph=False):
        """ 
        :attr:`backward` performs the following conceptual steps:

        1. fp32_loss = loss.float() (see first Note below)
        2. scaled_loss = fp32_loss*loss_scale
        3. scaled_loss.backward(), which accumulates scaled gradients into the ``.grad`` attributes of the model's leaves (which may be fp16, fp32, or a mixture, depending how your model was defined).
        4. fp16 grads are then copied to the master params' ``.grad`` attributes (see second Note), which are guaranteed to be fp32.
        5. Finally, master grads are divided by loss_scale.

        In this way, after :attr:`backward`, the master params have fresh gradients,
        and :attr:`step` may be called.

        .. note::
            :attr:`backward` internally converts the loss to fp32 before applying the loss scale.
            This provides some additional safety against overflow if the user has supplied an 
            fp16 loss value.  
            However, for maximum overflow safety, the user should
            compute the loss criterion (MSE, cross entropy, etc) in fp32 before supplying it to 
            :attr:`backward`.

        .. warning::
            The gradients found in a model's leaves after the call to 
            :attr:`backward` should not be regarded as valid in general, 
            because it's possible 
            they have been scaled (and in the case of dynamic loss scaling, 
            the scale factor may change over time).  
            If the user wants to inspect gradients after a call to :attr:`backward`,  
            only the master gradients should be regarded as valid.  These can be retrieved via
            :attr:`inspect_master_grad_data()`.

        Args:
            loss:  The loss output by the user's model.  loss may be either float or half (but see first Note above).
            update_master_grads (bool, optional, default=True):  Option to copy fp16 grads to fp32 grads on this call.  By setting this to False, the user can delay the copy, which is useful to eliminate redundant fp16->fp32 grad copies if :attr:`backward` is being called on multiple losses in one iteration.  If set to False, the user becomes responsible for calling :attr:`update_master_grads` before calling :attr:`step`.
            retain_graph (bool, optional, default=False):  Forwards the usual ``retain_graph=True`` option to the internal call to ``loss.backward``.  If ``retain_graph`` is being used to accumulate gradient values from multiple backward passes before calling ``optimizer.step``, passing ``update_master_grads=False`` is also recommended (see Example below).

        Example::

            # Ordinary operation:
            optimizer.backward(loss)

            # Naive operation with multiple losses (technically valid, but less efficient):
            # fp32 grads will be correct after the second call,  but 
            # the first call incurs an unnecessary fp16->fp32 grad copy.
            optimizer.backward(loss1)
            optimizer.backward(loss2)

            # More efficient way to handle multiple losses:
            # The fp16->fp32 grad copy is delayed until fp16 grads from all 
            # losses have been accumulated.
            optimizer.backward(loss1, update_master_grads=False)
            optimizer.backward(loss2, update_master_grads=False)
            optimizer.update_master_grads()
        """ 
        # To consider:  try multiple backward passes using retain_grad=True to find 
        # a loss scale that works.  After you find a loss scale that works, do a final dummy
        # backward pass with retain_graph=False to tear down the graph.  Doing this would avoid 
        # discarding the iteration,  but probably wouldn't improve overall efficiency.  
        scaled_loss = loss.float()*self.loss_scaler.loss_scale()
        scaled_loss.backward(retain_graph=retain_graph)
        if update_master_grads:
            self.update_master_grads()

    def update_master_grads(self):
        # torch.cuda.nvtx.range_push("update_master_grads")
        """
        Copy the ``.grad`` attribute from stored references to fp16 parameters to 
        the ``.grad`` attribute of the fp32 master parameters that are directly 
        updated by the optimizer.  :attr:`update_master_grads` only needs to be called if
        ``fp16_optimizer_obj.backward`` was called with ``update_master_grads=False``.
        """
        # if self.dynamic_loss_scale:
        #     self._check_overflow()
        #     if self.overflow: return
        # self._model_grads_to_master_grads()
        # self._downscale_master()
        # Use the one-shot multi-tensor apply kernel
        self.loss_scaler.clear_overflow_state()
        if len(self.all_fp16_params) > 0:
            # print("Model grads before")
            # print([param.grad.data for param in self.all_fp16_params])
            # I'm ONLY writing this as an incremental way to make some tests pass until
            # I can refactor the tests as well.
            # FP16_Optimizer should not be used by anyone.
            model_grads = []
            master_grads = []
            for model_param, master_param in zip(self.all_fp16_params,
                                                 self.all_fp32_from_fp16_params):
                if model_param.grad is not None:
                    model_grads.append(model_param.grad)
                    if master_param.grad is None:
                        master_param.grad = torch.empty_like(master_param)
                    master_grads.append(master_param.grad)
            self.loss_scaler.unscale(
                model_grads,
                master_grads,
                self.loss_scaler.loss_scale())
            # print("Master grads after")
            # print([param.grad.data for param in self.all_fp32_from_fp16_params])
        if len(self.all_fp32_from_fp32_params) > 0:
            model_grads = []
            master_grads = []
            for model_param, master_param in zip(self.all_fp32_from_fp32_params,
                                                 self.all_fp32_from_fp32_params):
                if model_param.grad is not None:
                    model_grads.append(model_param.grad)
                    master_grads.append(master_param.grad)
            # print("Model grads before")
            # print([param.grad.data for param in self.all_fp32_from_fp32_params])
            self.loss_scaler.unscale(
                model_grads,
                master_grads,
                self.loss_scaler.loss_scale())
            # print("Master grads after")
            # print([param.grad.data for param in self.all_fp32_from_fp32_params])
        # quit()
        self.overflow = self.loss_scaler.update_scale()
        # torch.cuda.nvtx.range_pop()


    def inspect_master_grad_data(self):
        """
        When running with :class:`FP16_Optimizer`, 
        ``.grad`` attributes of a model's fp16 leaves should not be
        regarded as truthful, because they might be scaled.  
        After a call to :attr:`fp16_optimizer_obj.backward(loss)`, if no overflow was encountered,
        the fp32 master params' ``.grad``
        attributes will contain valid gradients properly divided by the loss scale.  However, 
        because :class:`FP16_Optimizer` flattens some parameters, accessing them may be 
        nonintuitive.  :attr:`inspect_master_grad_data`
        allows those gradients to be viewed with shapes corresponding to their associated model leaves.

        Returns:
            List of lists (one list for each parameter group).  The list for each parameter group
            is a list of the ``.grad.data`` attributes of the fp32 master params belonging to that group.                 
        """
        if self.overflow:
            print("Warning:  calling FP16_Optimizer.inspect_master_grad_data while in an overflow state.  "
                  "Gradients are currently invalid (may be inf, nan, or stale).  Returning None.")
            return None
        else:
            # The optimizer owns only references to master params.
            master_grads_data = []
            for param_group in self.optimizer.param_groups:
                master_grads_this_group = []
                for param in param_group['params']:
                    if param.grad is not None:
                        master_grads_this_group.append(param.grad.data)
                    else:
                        master_grads_this_group.append(None)
                master_grads_data.append(master_grads_this_group)
            return master_grads_data


    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale()

    def _set_loss_scale(self, value):
        self.loss_scaler._loss_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)

    # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)
