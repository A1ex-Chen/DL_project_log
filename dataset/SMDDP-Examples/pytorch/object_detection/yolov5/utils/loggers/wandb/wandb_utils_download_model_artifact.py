def download_model_artifact(self, opt):
    """
        download the model checkpoint artifact if the resume path starts with WANDB_ARTIFACT_PREFIX

        arguments:
        opt (namespace) -- Commandline arguments for this run
        """
    if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
        model_artifact = wandb.use_artifact(remove_prefix(opt.resume,
            WANDB_ARTIFACT_PREFIX) + ':latest')
        assert model_artifact is not None, "Error: W&B model artifact doesn't exist"
        modeldir = model_artifact.download()
        total_epochs = model_artifact.metadata.get('total_epochs')
        is_finished = total_epochs is None
        assert not is_finished, 'training is finished, can only resume incomplete runs.'
        return modeldir, model_artifact
    return None, None
