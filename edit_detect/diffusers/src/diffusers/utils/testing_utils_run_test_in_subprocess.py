def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    To run a test in a subprocess. In particular, this can avoid (GPU) memory issue.

    Args:
        test_case (`unittest.TestCase`):
            The test that will run `target_func`.
        target_func (`Callable`):
            The function implementing the actual testing logic.
        inputs (`dict`, *optional*, defaults to `None`):
            The inputs that will be passed to `target_func` through an (input) queue.
        timeout (`int`, *optional*, defaults to `None`):
            The timeout (in seconds) that will be passed to the input and output queues. If not specified, the env.
            variable `PYTEST_TIMEOUT` will be checked. If still `None`, its value will be set to `600`.
    """
    if timeout is None:
        timeout = int(os.environ.get('PYTEST_TIMEOUT', 600))
    start_methohd = 'spawn'
    ctx = multiprocessing.get_context(start_methohd)
    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)
    input_queue.put(inputs, timeout=timeout)
    process = ctx.Process(target=target_func, args=(input_queue,
        output_queue, timeout))
    process.start()
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        process.terminate()
        test_case.fail(e)
    process.join(timeout=timeout)
    if results['error'] is not None:
        test_case.fail(f"{results['error']}")
