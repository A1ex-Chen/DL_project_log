def maybe_print(msg, rank0=False):
    if _amp_state.verbosity > 0:
        if rank0:
            if distributed:
                if torch.distributed.get_rank() == 0:
                    print(msg)
            else:
                print(msg)
        else:
            print(msg)
