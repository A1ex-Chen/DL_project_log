def log_value(self, step, key, val, stat='mean'):
    if self.enabled:
        if key not in self.cache:
            self.cache[key] = []
        self.cache[key].append(val)
        if len(self.cache[key]) == self.interval:
            agg_val = getattr(np, stat)(self.cache[key])
            self.summary_writer.add_scalar(key, agg_val, step)
            del self.cache[key]
