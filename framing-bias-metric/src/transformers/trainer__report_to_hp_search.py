def _report_to_hp_search(self, trial: Union['optuna.Trial', Dict[str, Any]],
    epoch: int, metrics: Dict[str, float]):
    if self.hp_search_backend is None or trial is None:
        return
    self.objective = self.compute_objective(metrics.copy())
    if self.hp_search_backend == HPSearchBackend.OPTUNA:
        trial.report(self.objective, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    elif self.hp_search_backend == HPSearchBackend.RAY:
        if self.state.global_step % self.args.save_steps == 0:
            self._tune_save_checkpoint()
        tune.report(objective=self.objective, **metrics)
