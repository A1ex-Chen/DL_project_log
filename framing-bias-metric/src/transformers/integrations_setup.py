def setup(self, args, state, model):
    """
        Setup the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.

                This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy
                whatever is in TrainerArgument's output_dir to the local or remote artifact storage. Using it without a
                remote storage will just copy the files to your artifact location.
        """
    log_artifacts = os.getenv('HF_MLFLOW_LOG_ARTIFACTS', 'FALSE').upper()
    if log_artifacts in {'TRUE', '1'}:
        self._log_artifacts = True
    if state.is_world_process_zero:
        mlflow.start_run()
        combined_dict = args.to_dict()
        if hasattr(model, 'config') and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}
        combined_dict_items = list(combined_dict.items())
        for i in range(0, len(combined_dict_items), MLflowCallback.MAX_LOG_SIZE
            ):
            mlflow.log_params(dict(combined_dict_items[i:i + MLflowCallback
                .MAX_LOG_SIZE]))
    self._initialized = True
