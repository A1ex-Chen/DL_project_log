def is_main_process() ->bool:
    return get_rank() == 0
