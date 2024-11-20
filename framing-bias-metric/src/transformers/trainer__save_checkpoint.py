def _save_checkpoint(self, model, trial, metrics=None):
    if hasattr(model, 'module'):
        assert model.module is self.model, f'Module {model.module} should be a reference to self.model'
    else:
        assert model is self.model, f'Model {model} should be a reference to self.model'
    checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
    if self.hp_search_backend is not None and trial is not None:
        run_id = (trial.number if self.hp_search_backend == HPSearchBackend
            .OPTUNA else tune.get_trial_id())
        run_name = self.hp_name(trial
            ) if self.hp_name is not None else f'run-{run_id}'
        output_dir = os.path.join(self.args.output_dir, run_name,
            checkpoint_folder)
    else:
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        self.store_flos()
    self.save_model(output_dir)
    if is_torch_tpu_available():
        xm.rendezvous('saving_optimizer_states')
        xm.save(self.optimizer.state_dict(), os.path.join(output_dir,
            'optimizer.pt'))
        with warnings.catch_warnings(record=True) as caught_warnings:
            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir,
                'scheduler.pt'))
            reissue_pt_warnings(caught_warnings)
    elif self.is_world_process_zero():
        torch.save(self.optimizer.state_dict(), os.path.join(output_dir,
            'optimizer.pt'))
        with warnings.catch_warnings(record=True) as caught_warnings:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(
                output_dir, 'scheduler.pt'))
        reissue_pt_warnings(caught_warnings)
    if metrics is not None and self.args.metric_for_best_model is not None:
        metric_to_check = self.args.metric_for_best_model
        if not metric_to_check.startswith('eval_'):
            metric_to_check = f'eval_{metric_to_check}'
        metric_value = metrics[metric_to_check]
        operator = np.greater if self.args.greater_is_better else np.less
        if (self.state.best_metric is None or self.state.
            best_model_checkpoint is None or operator(metric_value, self.
            state.best_metric)):
            self.state.best_metric = metric_value
            self.state.best_model_checkpoint = output_dir
    if self.is_world_process_zero():
        self.state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))
    if self.is_world_process_zero():
        self._rotate_checkpoints(use_mtime=True)
