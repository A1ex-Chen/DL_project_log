def _train_test_split(self, folders, json_names, val_size):
    if len(folders) > 0 and 'train' in folders and 'val' in folders:
        train_folder = os.path.join(self._json_dir, 'train/')
        train_json_names = [(train_sample_name + '.json') for
            train_sample_name in os.listdir(train_folder) if os.path.isdir(
            os.path.join(train_folder, train_sample_name))]
        val_folder = os.path.join(self._json_dir, 'val/')
        val_json_names = [(val_sample_name + '.json') for val_sample_name in
            os.listdir(val_folder) if os.path.isdir(os.path.join(val_folder,
            val_sample_name))]
        return train_json_names, val_json_names
    train_idxs, val_idxs = train_test_split(range(len(json_names)),
        test_size=val_size)
    train_json_names = [json_names[train_idx] for train_idx in train_idxs]
    val_json_names = [json_names[val_idx] for val_idx in val_idxs]
    return train_json_names, val_json_names
