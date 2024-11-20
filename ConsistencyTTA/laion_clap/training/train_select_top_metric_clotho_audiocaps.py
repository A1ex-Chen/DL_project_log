def select_top_metric_clotho_audiocaps(metrics, val_metrics_per_dataset, args):
    if not hasattr(args, 'top_selection_performance'):
        selection_performance = (
            calculate_selection_performance_clotho_audiocaps(
            val_metrics_per_dataset))
        metric_update = {}
        for n in val_metrics_per_dataset.keys():
            for k in val_metrics_per_dataset[n].keys():
                metric_update[k.split('/')[0] + '-top' + '/' + k.split('/')[1]
                    ] = val_metrics_per_dataset[n][k]
        metric_update['top_selection_performance'] = selection_performance
        metric_update['top-selection-epoch'] = metrics['epoch']
        metrics.update(metric_update)
        args.top_metric = metric_update
        args.top_selection_performance = selection_performance
    else:
        selection_performance_new = (
            calculate_selection_performance_clotho_audiocaps(
            val_metrics_per_dataset))
        selection_performance_old = args.top_selection_performance
        if selection_performance_new > selection_performance_old:
            metric_update = {}
            for n in val_metrics_per_dataset.keys():
                for k in val_metrics_per_dataset[n].keys():
                    metric_update[k.split('/')[0] + '-top' + '/' + k.split(
                        '/')[1]] = val_metrics_per_dataset[n][k]
            metric_update['top_selection_performance'
                ] = selection_performance_new
            metric_update['top-selection-epoch'] = metrics['epoch']
            metrics.update(metric_update)
            args.top_metric = metric_update
            args.top_selection_performance = selection_performance_new
        else:
            metrics.update(args.top_metric)
    return metrics
