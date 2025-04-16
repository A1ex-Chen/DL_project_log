def _create_loggers(self):
    root = Path(self.export_root)
    writer = SummaryWriter(root.joinpath('logs'))
    model_checkpoint = root.joinpath('models')
    train_loggers = [MetricGraphPrinter(writer, key='epoch', graph_name=
        'Epoch', group_name='Train'), MetricGraphPrinter(writer, key='loss',
        graph_name='Loss', group_name='Train')]
    val_loggers = []
    for k in self.metric_ks:
        val_loggers.append(MetricGraphPrinter(writer, key='NDCG@%d' % k,
            graph_name='NDCG@%d' % k, group_name='Validation'))
        val_loggers.append(MetricGraphPrinter(writer, key='Recall@%d' % k,
            graph_name='Recall@%d' % k, group_name='Validation'))
    val_loggers.append(RecentModelLogger(model_checkpoint))
    val_loggers.append(BestModelLogger(model_checkpoint, metric_key=self.
        best_metric))
    return writer, train_loggers, val_loggers
