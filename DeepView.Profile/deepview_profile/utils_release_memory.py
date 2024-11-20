def release_memory():
    logger.debug('Emptying cache')
    gc.collect()
    torch.cuda.empty_cache()
