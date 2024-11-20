def _tune_save_checkpoint(self):
    if not self.use_tune_checkpoints:
        return
    with tune.checkpoint_dir(step=self.state.global_step) as checkpoint_dir:
        self.args.output_dir = checkpoint_dir
        output_dir = os.path.join(self.args.output_dir,
            f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}')
        self.save_model(output_dir)
        if self.is_world_master():
            self.state.save_to_json(os.path.join(output_dir,
                'trainer_state.json'))
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir,
                'optimizer.pt'))
            torch.save(self.lr_scheduler.state_dict(), os.path.join(
                output_dir, 'scheduler.pt'))
