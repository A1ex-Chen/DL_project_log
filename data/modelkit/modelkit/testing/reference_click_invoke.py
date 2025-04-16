def click_invoke(runner, cmd_fn, args, env=None):
    res = runner.invoke(cmd_fn, args, env=env)
    if res.exception is not None and res.exc_info[0] != SystemExit:
        traceback.print_exception(*res.exc_info)
    return res
