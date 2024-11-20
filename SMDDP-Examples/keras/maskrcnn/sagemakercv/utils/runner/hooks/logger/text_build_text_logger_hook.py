@HOOKS.register('TextLoggerHook')
def build_text_logger_hook(cfg):
    return TextLoggerHook(interval=cfg.LOG_INTERVAL)
