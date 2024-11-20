def multi_process_func(*args, **kwargs):

    def wrapper_func(queue: Queue, *args):
        try:
            result = func(*args)
        except Exception as e:
            logger.error(e)
            print(e)
            result = 'N/A'
        queue.put(result)
    queue = Queue()
    p = Process(target=wrapper_func, args=[queue] + list(args))
    p.start()
    result = queue.get()
    p.join()
    return result
