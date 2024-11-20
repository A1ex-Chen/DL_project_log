@property
def summary_str(self):
    metric_str = []
    for name, meter in self.meters.items():
        metric_str.append('{}: {}'.format(name, meter.summary_str))
    return self.delimiter.join(metric_str)
