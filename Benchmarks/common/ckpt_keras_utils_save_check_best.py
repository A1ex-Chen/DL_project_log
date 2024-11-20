def save_check_best(self, logs, epoch):
    if not self.save_best:
        return False
    if self.save_best_metric not in logs.keys():
        raise Exception(('CandleCheckpointCallback: ' +
            "save_best_metric='%s' " + 'not in list of model metrics: %s') %
            (self.save_best_metric, str(logs.keys())))
    known_metrics = {'loss': '-', 'accuracy': '+', 'val_loss': '-',
        'val_accuracy': '+', 'lr': '-'}
    if self.save_best_metric not in known_metrics.keys():
        raise Exception(('CandleCheckpointCallback: ' +
            "save_best_metric='%s' " + 'not in list of known_metrics: %s') %
            (self.save_best_metric, str(known_metrics.keys())))
    if logs[self.save_best_metric] < self.best_metric_last:
        symbol = '<'
    elif logs[self.save_best_metric] > self.best_metric_last:
        symbol = '>'
    else:
        symbol = '='
    self.debug('metrics: %s: current=%f %s last=%f' % (self.
        save_best_metric, logs[self.save_best_metric], symbol, self.
        best_metric_last))
    improved = False
    if known_metrics[self.save_best_metric] == '-':
        if logs[self.save_best_metric] < self.best_metric_last:
            improved = True
    elif known_metrics[self.save_best_metric] == '+':
        if logs[self.save_best_metric] > self.best_metric_last:
            improved = True
    else:
        assert False
    if improved:
        self.best_metric_last = logs[self.save_best_metric]
        self.epoch_best = epoch
        return True
    return False
