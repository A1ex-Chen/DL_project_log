def dl_progress(count, block_size, total_size):
    global progbar
    if progbar is None:
        progbar = Progbar(total_size)
    else:
        progbar.update(count * block_size)
