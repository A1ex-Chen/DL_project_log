def is_master(args, local=False):
    return is_local_master(args) if local else is_global_master(args)
