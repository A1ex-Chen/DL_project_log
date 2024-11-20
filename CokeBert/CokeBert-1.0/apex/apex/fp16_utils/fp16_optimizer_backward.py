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
    scaled_loss = loss.float() * self.loss_scaler.loss_scale()
    scaled_loss.backward(retain_graph=retain_graph)
    if update_master_grads:
        self.update_master_grads()
