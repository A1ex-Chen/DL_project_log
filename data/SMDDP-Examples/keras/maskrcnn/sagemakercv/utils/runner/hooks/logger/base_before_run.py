def before_run(self, runner):
    for hook in runner.hooks[::-1]:
        if isinstance(hook, LoggerHook):
            hook.reset_flag = True
            break
