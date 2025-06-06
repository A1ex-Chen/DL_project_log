def prepare_data(data_args, model_args, training_args: TrainingArguments,
    preprocessor: Dict[str, Any]) ->Tuple[DatasetDict, Optional[ComputeMetrics]
    ]:
    datasets = {'train': partial(DATASETS.build, data_args.train) if
        training_args.do_train else None, 'validation': partial(DATASETS.
        build, data_args.validation) if training_args.do_eval else None,
        'test': partial(DATASETS.build, data_args.test) if training_args.
        do_predict else None}
    compute_metric_cfg = data_args.get('compute_metric', None)
    compute_metrics = build_compute_metric(compute_metric_cfg, preprocessor)
    conv_args = model_args.conv_args
    tokenize_kwargs = conv_args.get('tokenize_kwargs', {})
    conv_template = conv_args.get('conv_template', 'vicuna_v1.1')
    conv_template = partial(get_conv_template, name=conv_template)
    transforms = conv_args.get('transforms', None)
    if transforms is not None:
        transforms = TRANSFORMS.build(transforms)
    process_func = {}
    for k, v in model_args.process_func_args.items():
        process_func[k] = FUNCTIONS.build(cfg=v)
    conv_dataset_cls = partial(SingleImageConvDataset, preprocessor=
        preprocessor, process_func=process_func, tokenize_kwargs=
        tokenize_kwargs, conv_template=conv_template, training_args=
        training_args, transforms=transforms)
    ds = {'train': conv_dataset_cls(dataset_generator=datasets['train'],
        mode='train') if datasets['train'] is not None else None,
        'validation': conv_dataset_cls(dataset_generator=datasets[
        'validation'], mode='validation') if datasets['validation'] is not
        None else None, 'test': conv_dataset_cls(dataset_generator=datasets
        ['test'], mode='test') if datasets['test'] is not None else None}
    if hasattr(data_args, 'multitest') and bool(data_args.multitest
        ) and hasattr(training_args, 'do_multi_predict'
        ) and training_args.do_multi_predict:
        print(f'processing multitest set')
        k2v = {}
        for k, item in data_args.multitest.items():
            _dataset_cls = partial(DATASETS.build, item['cfg'])
            _compute_metric = build_compute_metric(item['compute_metric'],
                preprocessor)
            k2v[k] = {'dataset': conv_dataset_cls(dataset_generator=
                _dataset_cls, mode='test'), 'compute_metric': _compute_metric}
        ds['multitest'] = k2v
        print(f'processing multitest set. done.')
    lazy_init = data_args.get('lazy_init', True)
    if not lazy_init:
        init_ceph_client_if_needed()
    return ds, compute_metrics
