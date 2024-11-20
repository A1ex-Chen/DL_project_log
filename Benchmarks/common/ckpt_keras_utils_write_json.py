def write_json(self, jsonfile, epoch):
    from datetime import datetime
    now = datetime.now()
    D = {}
    D['epoch'] = epoch
    D['save_best_metric'] = self.save_best_metric
    D['best_metric_last'] = self.best_metric_last
    D['model_file'] = 'model.h5'
    D['checksum'] = self.cksum_model
    D['timestamp'] = now.strftime('%Y-%m-%d %H:%M:%S')
    if self.timestamp_last is None:
        time_elapsed = '__FIRST__'
    else:
        time_elapsed = (now - self.timestamp_last).total_seconds()
    self.timestamp_last = now
    D['time_elapsed'] = time_elapsed
    D['metadata'] = self.metadata
    with open(jsonfile, 'w') as fp:
        json.dump(D, fp)
        fp.write('\n')
