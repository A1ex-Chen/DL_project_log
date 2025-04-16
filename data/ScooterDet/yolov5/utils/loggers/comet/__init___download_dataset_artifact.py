def download_dataset_artifact(self, artifact_path):
    logged_artifact = self.experiment.get_artifact(artifact_path)
    artifact_save_dir = str(Path(self.opt.save_dir) / logged_artifact.name)
    logged_artifact.download(artifact_save_dir)
    metadata = logged_artifact.metadata
    data_dict = metadata.copy()
    data_dict['path'] = artifact_save_dir
    metadata_names = metadata.get('names')
    if isinstance(metadata_names, dict):
        data_dict['names'] = {int(k): v for k, v in metadata.get('names').
            items()}
    elif isinstance(metadata_names, list):
        data_dict['names'] = {int(k): v for k, v in zip(range(len(
            metadata_names)), metadata_names)}
    else:
        raise "Invalid 'names' field in dataset yaml file. Please use a list or dictionary"
    return self.update_data_paths(data_dict)
