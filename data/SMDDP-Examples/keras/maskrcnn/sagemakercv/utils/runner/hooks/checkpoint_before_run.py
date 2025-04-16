@master_only
def before_run(self, runner):
    if self.backbone_checkpoint:
        runner.logger.info('Loading checkpoint from %s...', self.
            backbone_checkpoint)
        chkp = tf.compat.v1.train.NewCheckpointReader(self.backbone_checkpoint)
        weights = [chkp.get_tensor(i) for i in ['/'.join(i.name.split('/')[
            -2:]).split(':')[0] for i in runner.trainer.model.layers[0].
            weights]]
        runner.trainer.model.layers[0].set_weights(weights)
        return
