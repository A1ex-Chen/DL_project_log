def print_once(msg):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(msg)
