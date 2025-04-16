def is_main_process():
    return get_rank() == 0
