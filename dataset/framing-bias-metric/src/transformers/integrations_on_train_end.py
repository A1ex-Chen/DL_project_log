def on_train_end(self, args, state, control, **kwargs):
    if self._initialized and state.is_world_process_zero:
        if self._log_artifacts:
            logger.info('Logging artifacts. This may take time.')
            mlflow.log_artifacts(args.output_dir)
        mlflow.end_run()
