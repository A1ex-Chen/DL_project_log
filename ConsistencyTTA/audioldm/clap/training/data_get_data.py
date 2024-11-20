def get_data(args, model_cfg):
    data = {}
    args.class_index_dict = load_class_label(args.class_label_path)
    if args.datasetinfos is None:
        args.datasetinfos = ['train', 'unbalanced_train', 'balanced_train']
    if args.dataset_type == 'webdataset':
        args.train_data = get_tar_path_from_dataset_name(args.datasetnames,
            args.datasetinfos, islocal=not args.remotedata, proportion=args
            .dataset_proportion, dataset_path=args.datasetpath,
            full_dataset=args.full_train_dataset)
        if args.full_train_dataset is None:
            args.full_train_dataset = []
        if args.exclude_eval_dataset is None:
            args.exclude_eval_dataset = []
        excluded_eval_datasets = (args.full_train_dataset + args.
            exclude_eval_dataset)
        val_dataset_names = [n for n in args.datasetnames if n not in
            excluded_eval_datasets
            ] if excluded_eval_datasets else args.datasetnames
        args.val_dataset_names = val_dataset_names
        args.val_data = get_tar_path_from_dataset_name(val_dataset_names, [
            'valid', 'test', 'eval'], islocal=not args.remotedata,
            proportion=1, dataset_path=args.datasetpath, full_dataset=None)
    if args.train_data:
        data['train'] = get_dataset_fn(args.train_data, args.dataset_type)(args
            , model_cfg, is_train=True)
    if args.val_data:
        data['val'] = get_dataset_fn(args.val_data, args.dataset_type)(args,
            model_cfg, is_train=False)
    return data
