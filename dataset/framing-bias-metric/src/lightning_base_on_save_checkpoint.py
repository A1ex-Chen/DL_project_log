@pl.utilities.rank_zero_only
def on_save_checkpoint(self, checkpoint: Dict[str, Any]) ->None:
    save_path = self.output_dir.joinpath('best_tfmr')
    self.model.config.save_step = self.step_count
    self.model.save_pretrained(save_path)
    self.tokenizer.save_pretrained(save_path)
