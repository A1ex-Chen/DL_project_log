def _initialize(models, optimizers, properties, num_losses=1,
    cast_model_outputs=None):
    from apex.parallel import DistributedDataParallel as apex_DDP
    from .amp import init as amp_init
    optimizers_was_list = False
    if isinstance(optimizers, torch.optim.Optimizer) or isinstance(optimizers,
        LARC):
        optimizers = [optimizers]
    elif optimizers is None:
        optimizers = []
    elif isinstance(optimizers, list):
        optimizers_was_list = True
        check_optimizers(optimizers)
    else:
        check_optimizers([optimizers])
        raise TypeError(
            'optimizers must be either a single optimizer or a list of optimizers.'
            )
    if isinstance(models, torch.nn.Module):
        models_was_list = False
        models = [models]
    elif isinstance(models, list):
        models_was_list = True
    else:
        raise TypeError(
            'models must be either a single model or a list of models.')
    check_models(models)
    if not _amp_state.allow_incoming_model_not_fp32:
        check_params_fp32(models)
    if properties.cast_model_type:
        if properties.keep_batchnorm_fp32:
            for model in models:
                convert_network(model, properties.cast_model_type)
        else:
            for model in models:
                model.to(properties.cast_model_type)
        input_caster = functools.partial(to_type, properties.cast_model_type)
        if cast_model_outputs is not None:
            output_caster = functools.partial(to_type, cast_model_outputs)
        else:
            output_caster = functools.partial(to_type, torch.float32)
        for model in models:

            def patch_forward(old_fwd):

                def new_fwd(*args, **kwargs):
                    output = old_fwd(*applier(args, input_caster), **
                        applier(kwargs, input_caster))
                    return applier(output, output_caster)
                return new_fwd
            model.forward = patch_forward(model.forward)
        for optimizer in optimizers:
            optimizer.load_state_dict(optimizer.state_dict())
    elif cast_model_outputs is not None:
        output_caster = functools.partial(to_type, cast_model_outputs)
        for model in models:

            def patch_forward(old_fwd):

                def new_fwd(*args, **kwargs):
                    output = old_fwd(*args, **kwargs)
                    return applier(output, output_caster)
                return new_fwd
            model.forward = patch_forward(model.forward)
    for i, optimizer in enumerate(optimizers):
        if isinstance(optimizer, FusedAdam):
            optimizers[i] = wrap_fused_adam(optimizer, properties)
        else:
            optimizers[i] = _process_optimizer(optimizer, properties)
    _amp_state.loss_scalers = []
    for _ in range(num_losses):
        _amp_state.loss_scalers.append(LossScaler(properties.loss_scale,
            min_loss_scale=_amp_state.min_loss_scale, max_loss_scale=
            _amp_state.max_loss_scale))
    if properties.patch_torch_functions:
        handle = amp_init(loss_scale=properties.loss_scale, verbose=
            _amp_state.verbosity == 2)
        for optimizer in optimizers:

            def patch_step(old_step):

                def new_step(*args, **kwargs):
                    with disable_casts():
                        output = old_step(*args, **kwargs)
                    return output
                return new_step
            optimizer.step = patch_step(optimizer.step)
    if optimizers_was_list:
        if models_was_list:
            return models, optimizers
        else:
            return models[0], optimizers
    elif models_was_list:
        if len(optimizers) == 0:
            return models
        else:
            return models, optimizers[0]
    elif len(optimizers) == 0:
        return models[0]
    else:
        return models[0], optimizers[0]
