def config_data_loaders(self):
    args = self.args
    self.dataloader_kwargs = {'timeout': 1, 'shuffle': 'True',
        'num_workers': NUM_WORKER if self.use_cuda else 0, 'pin_memory': 
        True if self.use_cuda else False}
    self.drug_resp_dataset_kwargs = {'data_root': DATA_ROOT, 'rand_state':
        args.rng_seed, 'summary': False, 'int_dtype': np.int8,
        'float_dtype': np.float16, 'output_dtype': np.float32,
        'grth_scaling': args.grth_scaling, 'dscptr_scaling': args.
        dscptr_scaling, 'rnaseq_scaling': args.rnaseq_scaling,
        'dscptr_nan_threshold': args.dscptr_nan_threshold,
        'rnaseq_feature_usage': args.rnaseq_feature_usage,
        'drug_feature_usage': args.drug_feature_usage, 'validation_ratio':
        args.val_split, 'disjoint_drugs': args.disjoint_drugs,
        'disjoint_cells': args.disjoint_cells}
    self.cl_clf_dataset_kwargs = {'data_root': DATA_ROOT, 'rand_state':
        args.rng_seed, 'summary': False, 'int_dtype': np.int8,
        'float_dtype': np.float16, 'output_dtype': np.float32,
        'rnaseq_scaling': args.rnaseq_scaling, 'rnaseq_feature_usage': args
        .rnaseq_feature_usage, 'validation_ratio': args.val_split}
    self.drug_target_dataset_kwargs = {'data_root': DATA_ROOT, 'rand_state':
        args.rng_seed, 'summary': False, 'int_dtype': np.int8,
        'float_dtype': np.float16, 'output_dtype': np.float32,
        'dscptr_scaling': args.dscptr_scaling, 'dscptr_nan_threshold': args
        .dscptr_nan_threshold, 'drug_feature_usage': args.
        drug_feature_usage, 'validation_ratio': args.val_split}
    self.drug_qed_dataset_kwargs = {'data_root': DATA_ROOT, 'rand_state':
        args.rng_seed, 'summary': False, 'int_dtype': np.int8,
        'float_dtype': np.float16, 'output_dtype': np.float32,
        'qed_scaling': args.qed_scaling, 'dscptr_scaling': args.
        dscptr_scaling, 'dscptr_nan_threshold': args.dscptr_nan_threshold,
        'drug_feature_usage': args.drug_feature_usage, 'validation_ratio':
        args.val_split}
