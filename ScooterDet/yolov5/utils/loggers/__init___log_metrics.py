def log_metrics(self, metrics, epoch):
    if self.csv:
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 1
        s = '' if self.csv.exists() else ('%23s,' * n % tuple(['epoch'] + keys)
            ).rstrip(',') + '\n'
        with open(self.csv, 'a') as f:
            f.write(s + ('%23.5g,' * n % tuple([epoch] + vals)).rstrip(',') +
                '\n')
    if self.tb:
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, epoch)
    if self.wandb:
        self.wandb.log(metrics, step=epoch)
