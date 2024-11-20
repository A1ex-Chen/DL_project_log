def download_dataset_artifact(self, path, alias):
    """
        download the model checkpoint artifact if the path starts with WANDB_ARTIFACT_PREFIX

        arguments:
        path -- path of the dataset to be used for training
        alias (str)-- alias of the artifact to be download/used for training

        returns:
        (str, wandb.Artifact) -- path of the downladed dataset and it's corresponding artifact object if dataset
        is found otherwise returns (None, None)
        """
    if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
        artifact_path = Path(remove_prefix(path, WANDB_ARTIFACT_PREFIX) +
            ':' + alias)
        dataset_artifact = wandb.use_artifact(artifact_path.as_posix().
            replace('\\', '/'))
        assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn't exist'"
        datadir = dataset_artifact.download()
        return datadir, dataset_artifact
    return None, None
