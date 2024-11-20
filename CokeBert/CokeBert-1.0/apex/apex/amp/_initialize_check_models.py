def check_models(models):
    for model in models:
        parallel_type = None
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            parallel_type = 'torch.nn.parallel.DistributedDataParallel'
        if isinstance(model, apex_DDP):
            parallel_type = 'apex.parallel.DistributedDataParallel'
        if isinstance(model, torch.nn.parallel.DataParallel):
            parallel_type = 'torch.nn.parallel.DataParallel'
        if parallel_type is not None:
            raise RuntimeError('Incoming model is an instance of {}. '.
                format(parallel_type) +
                """Parallel wrappers should only be applied to the model(s) AFTER 
the model(s) have been returned from amp.initialize."""
                )
