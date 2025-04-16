def step(self, closure=None):
    """
        If no closure is supplied, :attr:`step` should be called after 
        ``fp16_optimizer_obj.backward(loss)``.
        :attr:`step` updates the fp32 master copy of parameters using the optimizer supplied to
        :class:`FP16_Optimizer`'s constructor, then copies the updated fp32 params into the fp16 params
        originally referenced by :class:`FP16_Optimizer`'s constructor, so the user may immediately run
        another forward pass using their model.

        If a closure is supplied, :attr:`step` may be called without a prior call to 
        :attr:`backward(loss)`.
        This control flow is identical to `ordinary Pytorch optimizer use`_ with closures.
        However, the user should take care that any ``loss.backward()`` call within the closure
        has been replaced by ``fp16_optimizer_obj.backward(loss)``.

        Args:
           closure (optional):  Closure that will be supplied to the underlying optimizer originally passed to :class:`FP16_Optimizer`'s constructor.  closure should call :attr:`zero_grad()` on the :class:`FP16_Optimizer` object, compute the loss, call :attr:`backward(loss)`, and return the loss.

        Example with closure::

            # optimizer is assumed to be an FP16_Optimizer object, previously constructed from an 
            # existing pytorch optimizer.
            for input, target in dataset:
                def closure():
                    optimizer.zero_grad()
                    output = model(input)
                    loss = loss_fn(output, target)
                    # loss.backward() becomes:
                    optimizer.backward(loss)
                    return loss
                optimizer.step(closure)

        .. warning::
            Currently, calling :attr:`step` with a closure is not compatible with dynamic loss scaling.

        .. _`ordinary Pytorch optimizer use`:
            http://pytorch.org/docs/master/optim.html#optimizer-step-closure
        """
    scale = self.loss_scaler.loss_scale()
    if self.overflow:
        maybe_print('Gradient overflow.  Skipping step, reducing ' +
            'loss scale to {}'.format(self.loss_scaler.loss_scale()))
        return
    if closure is not None:
        retval = self._step_with_closure(closure)
    else:
        retval = self.optimizer.step()
    self._master_params_to_model_params()
    return retval
