def upload_dataset_artifact(self):
    dataset_name = self.data_dict.get('dataset_name', 'yolov5-dataset')
    path = str((ROOT / Path(self.data_dict['path'])).resolve())
    metadata = self.data_dict.copy()
    for key in ['train', 'val', 'test']:
        split_path = metadata.get(key)
        if split_path is not None:
            metadata[key] = split_path.replace(path, '')
    artifact = comet_ml.Artifact(name=dataset_name, artifact_type='dataset',
        metadata=metadata)
    for key in metadata.keys():
        if key in ['train', 'val', 'test']:
            if isinstance(self.upload_dataset, str
                ) and key != self.upload_dataset:
                continue
            asset_path = self.data_dict.get(key)
            if asset_path is not None:
                artifact = self.add_assets_to_artifact(artifact, path,
                    asset_path, key)
    self.experiment.log_artifact(artifact)
    return
