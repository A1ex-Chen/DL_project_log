def register_metric(self, metric_name, meter, verbosity=0, metadata={}):
    if self.verbose:
        print('Registering metric: {}'.format(metric_name))
    self.metrics[metric_name] = {'meter': meter, 'level': verbosity}
    dllogger.metadata(metric_name, metadata)
