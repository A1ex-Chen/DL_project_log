def _save_checkpoint(self, model, trial, metrics=None):
    if getattr(self.args, 'tune_mm_mlp_adapter', False):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        keys_to_match = ['mm_projector', 'vision_resampler']
        if getattr(self.args, 'use_im_start_end', False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])
        weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.
            named_parameters(), keys_to_match)
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            self.model.config.save_pretrained(output_dir)
            torch.save(weight_to_save, os.path.join(output_dir,
                f'mm_projector.bin'))
    else:
        super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)
