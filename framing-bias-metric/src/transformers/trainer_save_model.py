def save_model(self, output_dir: Optional[str]=None):
    """
        Will save the model, so you can reload it using :obj:`from_pretrained()`.

        Will only save from the world_master process (unless in TPUs).
        """
    if is_torch_tpu_available():
        self._save_tpu(output_dir)
    elif self.is_world_process_zero():
        self._save(output_dir)
