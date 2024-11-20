def resume_from_checkpoint(self, model_dir: str, train_steps: int) ->int:
    """Resumes from the latest checkpoint, if possible.

        Loads the model weights and optimizer settings from a checkpoint.
        This function should be used in case of preemption recovery.

        Args:
        model: The model whose weights should be restored.
        model_dir: The directory where model weights were saved.
        train_steps: The number of steps to train.

        Returns:
        The epoch of the latest checkpoint, or 0 if not restoring.

        """
    last_iteration = self.load_model_weights(model_dir)
    initial_epoch = last_iteration // train_steps
    return int(initial_epoch)
