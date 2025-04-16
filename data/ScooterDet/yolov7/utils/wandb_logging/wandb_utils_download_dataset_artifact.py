def download_dataset_artifact(self, path, alias):
    if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
        dataset_artifact = wandb.use_artifact(remove_prefix(path,
            WANDB_ARTIFACT_PREFIX) + ':' + alias)
        assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn't exist'"
        datadir = dataset_artifact.download()
        return datadir, dataset_artifact
    return None, None
