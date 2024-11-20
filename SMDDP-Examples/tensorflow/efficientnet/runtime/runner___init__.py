def __init__(self, flags, logger):
    self.params = flags
    self.logger = logger
    if sdp.rank() == 0:
        self.serialize_config(model_dir=self.params.model_dir)
    label_smoothing = flags.label_smoothing
    self.one_hot = label_smoothing and label_smoothing > 0
    builders = get_dataset_builders(self.params, self.one_hot)
    datasets = [(builder.build() if builder else None) for builder in builders]
    self.train_dataset, self.validation_dataset = datasets
    self.train_builder, self.validation_builder = builders
    self.initialize()
    model_params = build_model_params(model_name=self.params.arch,
        is_training='predict' not in self.params.mode, batch_norm=self.
        params.batch_norm, num_classes=self.params.num_classes, activation=
        self.params.activation, dtype=DTYPE_MAP[self.params.dtype],
        weight_decay=self.params.weight_decay, weight_init=self.params.
        weight_init)
    models_dict = get_models()
    self.model = [model for model_name, model in models_dict.items() if 
        model_name in self.params.arch][0](**model_params)
    self.metrics = ['accuracy', 'top_5']
    if self.params.dataset == 'ImageNet':
        self.train_num_examples = 1281167
        self.eval_num_examples = 50000
