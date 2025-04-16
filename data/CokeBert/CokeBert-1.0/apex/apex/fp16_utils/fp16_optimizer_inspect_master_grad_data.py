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
        print(
            'Warning:  calling FP16_Optimizer.inspect_master_grad_data while in an overflow state.  Gradients are currently invalid (may be inf, nan, or stale).  Returning None.'
            )
        return None
    else:
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
