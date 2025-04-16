def wrapper_func(queue: Queue, *args):
    try:
        result = func(*args)
    except Exception as e:
        logger.error(e)
        print(e)
        result = 'N/A'
    queue.put(result)
