def model_grads_to_master_grads(model_params, master_params, flat_master=False
    ):
    """
    Copy model gradients to master gradients.  

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`model_grads_to_master_grads`.
    """
    if flat_master:
        master_params[0].grad.data.copy_(_flatten_dense_tensors([p.grad.
            data for p in model_params]))
    else:
        for model, master in zip(model_params, master_params):
            if model.grad is not None:
                if master.grad is None:
                    master.grad = Variable(master.data.new(*master.data.size())
                        )
                master.grad.data.copy_(model.grad.data)
            else:
                master.grad = None
