def log_model(self, model_path, epoch=0, metadata={}):
    if self.wandb:
        art = wandb.Artifact(name=f'run_{wandb.run.id}_model', type='model',
            metadata=metadata)
        art.add_file(str(model_path))
        wandb.log_artifact(art)
