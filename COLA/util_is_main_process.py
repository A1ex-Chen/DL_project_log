def is_main_process(local_rank):
    return local_rank == 0 or local_rank == -1
