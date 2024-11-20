def __str__(self):
    metric_str = []
    for name, meter in self.meters.items():
        metric_str.append('{}: {}'.format(name, str(meter)))
    return self.delimiter.join(metric_str)
