def separate_process_wrapper_fn(func: Callable[[], None],
    do_multi_processing: bool) ->Callable[[], None]:
    """
    This function wraps another function into its own separated process. In order to ensure accurate memory
    measurements it is important that the function is executed in a separate process

    Args:

        - `func`: (`callable`): function() -> ... generic function which will be executed in its own separate process
        - `do_multi_processing`: (`bool`) Whether to run function on separate process or not
    """

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
    if do_multi_processing:
        logger.info(f'Function {func} is executed in its own process...')
        return multi_process_func
    else:
        return func
