def build_metric(metric_type, dataset_name, **kwargs):
    build_func = evaluation_protocol[metric_type][dataset_name]
    return build_func(dataset_name=dataset_name, **kwargs)
