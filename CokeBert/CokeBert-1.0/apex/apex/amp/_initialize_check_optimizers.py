def check_optimizers(optimizers):
    for optim in optimizers:
        bad_optim_type = None
        if isinstance(optim, FP16_Optimizer_general):
            bad_optim_type = 'apex.fp16_utils.FP16_Optimizer'
        if isinstance(optim, FP16_Optimizer_for_fused):
            bad_optim_type = 'apex.optimizers.FP16_Optimizer'
        if bad_optim_type is not None:
            raise RuntimeError(
                'An incoming optimizer is an instance of {}. '.format(
                bad_optim_type) +
                """The optimizer(s) passed to amp.initialize() must be bare 
instances of either ordinary Pytorch optimizers, or Apex fused 
optimizers (currently just FusedAdam, but FusedSGD will be added 
soon).  You should not manually wrap your optimizer in either 
apex.fp16_utils.FP16_Optimizer or apex.optimizers.FP16_Optimizer. 
amp.initialize will take care of that for you (if necessary) based 
on the specified opt_level (and optional overridden properties)."""
                )
