def is_local_master(args):
    return args.local_rank == 0
