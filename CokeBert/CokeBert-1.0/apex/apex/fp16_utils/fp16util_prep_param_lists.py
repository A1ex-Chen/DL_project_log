def prep_param_lists(model, flat_master=False):
    """
    Creates a list of FP32 master parameters for a given model, as in
    `Training Neural Networks with Mixed Precision:  Real Examples`_.

    Args:
        model (torch.nn.Module): Existing Pytorch model
        flat_master (bool, optional, default=False):  Flatten the master parameters into a single tensor, as a performance optimization.
    Returns:
        A tuple (``model_params``, ``master_params``). ``model_params`` is a list of the model's parameters for later use with :func:`model_grads_to_master_grads` and :func:`master_params_to_model_params`.  ``master_params`` is a list of FP32 master gradients.  If ``flat_master=True``, ``master_params`` will be a list with one element.

    Example::

        model_params, master_params = prep_param_lists(model)

    .. warning::
        Currently, if ``flat_master=True``, all the model's parameters must be the same type.  If the model has parameters of different types, use ``flat_master=False``, or use :class:`FP16_Optimizer`.

    .. _`Training Neural Networks with Mixed Precision:  Real Examples`:
        http://on-demand.gputechconf.com/gtc/2018/video/S81012/
    """
    model_params = [param for param in model.parameters() if param.
        requires_grad]
    if flat_master:
        try:
            master_params = _flatten_dense_tensors([param.data for param in
                model_params]).float()
        except:
            print(
                'Error in prep_param_lists:  model may contain a mixture of parameters of different types.  Use flat_master=False, or use F16_Optimizer.'
                )
            raise
        master_params = torch.nn.Parameter(master_params)
        master_params.requires_grad = True
        if master_params.grad is None:
            master_params.grad = master_params.new(*master_params.size())
        return model_params, [master_params]
    else:
        master_params = [param.clone().float().detach() for param in
            model_params]
        for param in master_params:
            param.requires_grad = True
        return model_params, master_params
