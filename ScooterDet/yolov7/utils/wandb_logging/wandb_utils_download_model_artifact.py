def download_model_artifact(self, opt):
    if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
        model_artifact = wandb.use_artifact(remove_prefix(opt.resume,
            WANDB_ARTIFACT_PREFIX) + ':latest')
        assert model_artifact is not None, "Error: W&B model artifact doesn't exist"
        modeldir = model_artifact.download()
        epochs_trained = model_artifact.metadata.get('epochs_trained')
        total_epochs = model_artifact.metadata.get('total_epochs')
        assert epochs_trained < total_epochs, 'training to %g epochs is finished, nothing to resume.' % total_epochs
        return modeldir, model_artifact
    return None, None
