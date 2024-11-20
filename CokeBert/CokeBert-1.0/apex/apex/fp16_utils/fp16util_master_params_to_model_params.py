def master_params_to_model_params(model_params, master_params, flat_master=
    False):
    """
    Copy master parameters to model parameters.

    Args:
        model_params:  List of model parameters created by :func:`prep_param_lists`.
        master_params:  List of FP32 master parameters created by :func:`prep_param_lists`.  If ``master_params`` was created with ``flat_master=True``, ``flat_master=True`` should also be supplied to :func:`master_params_to_model_params`.
    """
    if flat_master:
        for model, master in zip(model_params, _unflatten_dense_tensors(
            master_params[0].data, model_params)):
            model.data.copy_(master)
    else:
        for model, master in zip(model_params, master_params):
            model.data.copy_(master.data)
