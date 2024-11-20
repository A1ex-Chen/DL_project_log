def rank0_print(*args):
    if local_rank == 0:
        print(*args)
