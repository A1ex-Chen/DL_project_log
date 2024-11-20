def log_model(self, path, opt, epoch, fitness_score, best_model=False):
    """
        Log the model checkpoint as W&B artifact.

        arguments:
        path (Path)   -- Path of directory containing the checkpoints
        opt (namespace) -- Command line arguments for this run
        epoch (int)  -- Current epoch number
        fitness_score (float) -- fitness score for current epoch
        best_model (boolean) -- Boolean representing if the current checkpoint is the best yet.
        """
    model_artifact = wandb.Artifact(f'run_{wandb.run.id}_model', type=
        'model', metadata={'original_url': str(path), 'epochs_trained': 
        epoch + 1, 'save period': opt.save_period, 'project': opt.project,
        'total_epochs': opt.epochs, 'fitness_score': fitness_score})
    model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
    wandb.log_artifact(model_artifact, aliases=['latest', 'last',
        f'epoch {str(self.current_epoch)}', 'best' if best_model else ''])
    LOGGER.info(f'Saving model artifact on epoch {epoch + 1}')
