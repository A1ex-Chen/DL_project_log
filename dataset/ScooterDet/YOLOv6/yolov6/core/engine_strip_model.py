def strip_model(self):
    if self.main_process:
        LOGGER.info(
            f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.'
            )
        save_ckpt_dir = osp.join(self.save_dir, 'weights')
        strip_optimizer(save_ckpt_dir, self.epoch)
