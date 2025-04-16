def get_system_metrics(self):
    system_stats = dict()
    system_stats['cpu_percent'] = psutil.cpu_percent()
    system_stats.update(self.gpu_stats())
    system_stats['disk_util'] = psutil.disk_usage('/').percent
    system_stats['mem_util'] = psutil.virtual_memory().percent
    return system_stats
