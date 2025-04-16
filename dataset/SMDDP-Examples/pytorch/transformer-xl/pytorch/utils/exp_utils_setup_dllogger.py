def setup_dllogger(enabled=True, filename=os.devnull):
    rank = utils.distributed.get_rank()
    if enabled and rank == 0:
        backends = [dllogger.JSONStreamBackend(dllogger.Verbosity.VERBOSE,
            filename)]
        dllogger.init(backends)
    else:
        dllogger.init([])
