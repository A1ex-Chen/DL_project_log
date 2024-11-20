def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(key, get_lvis_instances_meta(
                dataset_name), os.path.join(root, json_file) if '://' not in
                json_file else json_file, os.path.join(root, image_root))
