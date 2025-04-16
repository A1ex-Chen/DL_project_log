def __init__(self, collection_interval=5, record_interval=50,
    rolling_mean_interval=12):
    self.running = False
    self.collection_interval = collection_interval
    self.record_interval = record_interval
    self.rolling_mean_interval = rolling_mean_interval
    self.system_metrics = defaultdict(list)
