def __init__(self, model, args):
    super(DeepSpeedAgent, self).__init__()
    self.args = args
    self.model = model
    self.writer = SummaryWriter(args['log_path'])
    ds_params = args['deepspeed']
    ds_params['scheduler']['params']['total_num_steps'] = self.args[
        'total_steps']
    ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self
        .args['total_steps'] * self.args['warmup_rate']))
    self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(model=self.
        model, model_parameters=self.model.parameters(), config_params=
        ds_params, dist_init_required=True, args=types.SimpleNamespace(**args))
