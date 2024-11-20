def bar_update(count, block_size, total_size):
    if pbar.total is None and total_size:
        pbar.total = total_size
    progress_bytes = count * block_size
    pbar.update(progress_bytes - pbar.n)
