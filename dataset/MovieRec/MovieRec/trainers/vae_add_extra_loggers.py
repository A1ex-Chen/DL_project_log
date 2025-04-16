def add_extra_loggers(self):
    cur_beta_logger = MetricGraphPrinter(self.writer, key='cur_beta',
        graph_name='Beta', group_name='Train')
    self.train_loggers.append(cur_beta_logger)
    if self.args.find_best_beta:
        best_beta_logger = MetricGraphPrinter(self.writer, key='best_beta',
            graph_name='Best_beta', group_name='Validation')
        self.val_loggers.append(best_beta_logger)
