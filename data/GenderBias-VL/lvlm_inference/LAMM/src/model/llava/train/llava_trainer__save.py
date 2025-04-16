def _save(self, output_dir: Optional[str]=None, state_dict=None):
    if getattr(self.args, 'tune_mm_mlp_adapter', False):
        pass
    else:
        super(LLaVATrainer, self)._save(output_dir, state_dict)
