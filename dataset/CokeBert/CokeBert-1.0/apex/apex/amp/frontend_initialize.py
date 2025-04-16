def initialize(models, optimizers=None, enabled=True, opt_level='O1',
    cast_model_type=None, patch_torch_functions=None, keep_batchnorm_fp32=
    None, master_weights=None, loss_scale=None, cast_model_outputs=None,
    num_losses=1, verbosity=1, min_loss_scale=None, max_loss_scale=2.0 ** 24):
    """
    Initialize your models, optimizers, and the Torch tensor and functional namespace according to the
    chosen ``opt_level`` and overridden properties, if any.

    ``amp.initialize`` should be called **after** you have finished
    constructing your model(s) and
    optimizer(s), but **before** you send your model through any DistributedDataParallel wrapper.
    See `Distributed training`_ in the Imagenet example.

    Currently, ``amp.initialize`` should only be called **once**,
    although it can process an arbitrary number of
    models and optimizers (see the corresponding `Advanced Amp Usage topic`_).
    If you think your use case requires ``amp.initialize`` to be called more than once,
    `let us know`_.

    Any property keyword argument that is not ``None`` will be interpreted as a manual override.

    To prevent having to rewrite anything else in your script, name the returned models/optimizers
    to replace the passed models/optimizers, as in the code sample below.

    Args:
        models (torch.nn.Module or list of torch.nn.Modules):  Models to modify/cast.
        optimizers (optional, torch.optim.Optimizer or list of torch.optim.Optimizers):  Optimizers to modify/cast.
            REQUIRED for training, optional for inference.
        enabled (bool, optional, default=True):  If False, renders all Amp calls no-ops, so your script
            should run as if Amp were not present.
        opt_level (str, optional, default="O1"):  Pure or mixed precision optimization level.  Accepted values are
            "O0", "O1", "O2", and "O3", explained in detail above.
        cast_model_type (``torch.dtype``, optional, default=None):  Optional property override, see
            above.
        patch_torch_functions (bool, optional, default=None):  Optional property override.
        keep_batchnorm_fp32 (bool or str, optional, default=None):  Optional property override.  If
            passed as a string, must be the string "True" or "False".
        master_weights (bool, optional, default=None):  Optional property override.
        loss_scale (float or str, optional, default=None):  Optional property override.  If passed as a string,
            must be a string representing a number, e.g., "128.0", or the string "dynamic".
        cast_model_outputs (torch.dtype, optional, default=None):  Option to ensure that the outputs
            of your model(s) are always cast to a particular type regardless of ``opt_level``.
        num_losses (int, optional, default=1):  Option to tell Amp in advance how many losses/backward
            passes you plan to use.  When used in conjunction with the ``loss_id`` argument to
            ``amp.scale_loss``, enables Amp to use a different loss scale per loss/backward pass,
            which can improve stability.  See "Multiple models/optimizers/losses"
            under `Advanced Amp Usage`_ for examples.  If ``num_losses`` is left to 1, Amp will still
            support multiple losses/backward passes, but use a single global loss scale
            for all of them.
        verbosity (int, default=1):  Set to 0 to suppress Amp-related output.
        min_loss_scale (float, default=None):  Sets a floor for the loss scale values that can be chosen by dynamic
            loss scaling.  The default value of None means that no floor is imposed.
            If dynamic loss scaling is not used, `min_loss_scale` is ignored.
        max_loss_scale (float, default=2.**24):  Sets a ceiling for the loss scale values that can be chosen by
            dynamic loss scaling.  If dynamic loss scaling is not used, `max_loss_scale` is ignored.

    Returns:
        Model(s) and optimizer(s) modified according to the ``opt_level``.
        If either the ``models`` or ``optimizers`` args were lists, the corresponding return value will
        also be a list.

    Permissible invocations::

        model, optim = amp.initialize(model, optim,...)
        model, [optim1, optim2] = amp.initialize(model, [optim1, optim2],...)
        [model1, model2], optim = amp.initialize([model1, model2], optim,...)
        [model1, model2], [optim1, optim2] = amp.initialize([model1, model2], [optim1, optim2],...)

        # This is not an exhaustive list of the cross product of options that are possible,
        # just a set of examples.
        model, optim = amp.initialize(model, optim, opt_level="O0")
        model, optim = amp.initialize(model, optim, opt_level="O0", loss_scale="dynamic"|128.0|"128.0")

        model, optim = amp.initialize(model, optim, opt_level="O1") # uses "loss_scale="dynamic" default
        model, optim = amp.initialize(model, optim, opt_level="O1", loss_scale=128.0|"128.0")

        model, optim = amp.initialize(model, optim, opt_level="O2") # uses "loss_scale="dynamic" default
        model, optim = amp.initialize(model, optim, opt_level="O2", loss_scale=128.0|"128.0")
        model, optim = amp.initialize(model, optim, opt_level="O2", keep_batchnorm_fp32=True|False|"True"|"False")

        model, optim = amp.initialize(model, optim, opt_level="O3") # uses loss_scale=1.0 default
        model, optim = amp.initialize(model, optim, opt_level="O3", loss_scale="dynamic"|128.0|"128.0")
        model, optim = amp.initialize(model, optim, opt_level="O3", keep_batchnorm_fp32=True|False|"True"|"False")

    The `Imagenet example`_ demonstrates live use of various opt_levels and overrides.

    .. _`Distributed training`:
        https://github.com/NVIDIA/apex/tree/master/examples/imagenet#distributed-training

    .. _`Imagenet example`:
        https://github.com/NVIDIA/apex/tree/master/examples/imagenet

    .. _`Advanced Amp Usage`:
        https://nvidia.github.io/apex/advanced.html

    .. _`Advanced Amp Usage topic`:
        https://nvidia.github.io/apex/advanced.html#multiple-models-optimizers-losses

    .. _`let us know`:
        https://github.com/NVIDIA/apex/issues
    """
    _amp_state.opt_properties = Properties()
    _amp_state.verbosity = verbosity
    if not enabled:
        if optimizers is None:
            return models
        else:
            return models, optimizers
    if not torch.backends.cudnn.enabled:
        raise RuntimeError('Amp requires torch.backends.cudnn.enabled = True')
    if opt_level not in opt_levels:
        raise RuntimeError('Unexpected optimization level {}. '.format(
            opt_level) +
            "Options are 'O0', 'O1', 'O2', 'O3'.  Note that in `O0`, `O1`, etc., the prefix O is the letter O, "
             + 'not the number zero.')
    else:
        _amp_state.opt_properties = opt_levels[opt_level](_amp_state.
            opt_properties)
        maybe_print('Selected optimization level {}'.format(opt_levels[
            opt_level].brief), True)
        maybe_print('Defaults for this optimization level are:', True)
        for k, v in _amp_state.opt_properties.options.items():
            maybe_print('{:22} : {}'.format(k, v), True)
    _amp_state.min_loss_scale = min_loss_scale
    _amp_state.max_loss_scale = max_loss_scale
    maybe_print(
        'Processing user overrides (additional kwargs that are not None)...',
        True)
    if enabled is not None:
        _amp_state.opt_properties.enabled = enabled
    if opt_level is not None:
        _amp_state.opt_properties.opt_level = opt_level
    if cast_model_type is not None:
        _amp_state.opt_properties.cast_model_type = cast_model_type
    if patch_torch_functions is not None:
        _amp_state.opt_properties.patch_torch_functions = patch_torch_functions
    if keep_batchnorm_fp32 is not None:
        _amp_state.opt_properties.keep_batchnorm_fp32 = keep_batchnorm_fp32
    if master_weights is not None:
        _amp_state.opt_properties.master_weights = master_weights
    if loss_scale is not None:
        _amp_state.opt_properties.loss_scale = loss_scale
    maybe_print('After processing overrides, optimization options are:', True)
    for k, v in _amp_state.opt_properties.options.items():
        maybe_print('{:22} : {}'.format(k, v), True)
    return _initialize(models, optimizers, _amp_state.opt_properties,
        num_losses, cast_model_outputs)
