def build_hooks(self):
    """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
    cfg = self.cfg.clone()
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = 0
    ret = [hooks.IterationTimer(), hooks.LRScheduler(), hooks.PreciseBN(cfg
        .TEST.EVAL_PERIOD, self.model, self.build_train_loader(cfg), cfg.
        TEST.PRECISE_BN.NUM_ITER) if cfg.TEST.PRECISE_BN.ENABLED and
        get_bn_modules(self.model) else None]
    if comm.is_main_process():
        ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER
            .CHECKPOINT_PERIOD))

    def test_and_save_results():
        self._last_eval_results = self.test(self.cfg, self.model)
        return self._last_eval_results
    ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results,
        eval_after_train=cfg.TEST.EVAL_AFTER_TRAIN))
    if comm.is_main_process():
        ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
    return ret
