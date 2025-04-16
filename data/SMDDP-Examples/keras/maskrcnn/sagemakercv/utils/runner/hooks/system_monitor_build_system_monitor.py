@HOOKS.register('SystemMonitor')
def build_system_monitor(cfg):
    return SystemMonitor()
