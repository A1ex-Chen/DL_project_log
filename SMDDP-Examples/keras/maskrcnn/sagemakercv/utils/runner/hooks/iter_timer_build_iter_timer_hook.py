@HOOKS.register('IterTimerHook')
def build_iter_timer_hook(cfg):
    return IterTimerHook(interval=cfg.LOG_INTERVAL)
