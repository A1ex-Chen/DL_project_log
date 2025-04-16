def save_metrics(self, metrics):
    """Saves training metrics to a CSV file."""
    keys, vals = list(metrics.keys()), list(metrics.values())
    n = len(metrics) + 1
    s = '' if self.csv.exists() else ('%23s,' * n % tuple(['epoch'] + keys)
        ).rstrip(',') + '\n'
    with open(self.csv, 'a') as f:
        f.write(s + ('%23.5g,' * n % tuple([self.epoch + 1] + vals)).rstrip
            (',') + '\n')
