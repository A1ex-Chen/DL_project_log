def on_log(self, args, state, control, logs, model=None, **kwargs):
    if not self._initialized:
        self.setup(args, state, model)
    if state.is_world_process_zero:
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=state.global_step)
            else:
                logger.warning(
                    'Trainer is attempting to log a value of "%s" of type %s for key "%s" as a metric. MLflow\'s log_metric() only accepts float and int types so we dropped this attribute.'
                    , v, type(v), k)
