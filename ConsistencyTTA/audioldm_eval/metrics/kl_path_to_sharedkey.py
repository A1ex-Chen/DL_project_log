def path_to_sharedkey(path, dataset_name, classes=None):
    if dataset_name.lower() == 'vggsound':
        sharedkey = Path(path).stem.replace('_mel', '').split('_sample_')[0]
    elif dataset_name.lower() == 'vas':
        classes = sorted(classes)
        target_to_label = {f'cls_{i}': c for i, c in enumerate(classes)}
        for folder_cls_name, label in target_to_label.items():
            path = path.replace(folder_cls_name, label).replace(
                'melspec_10s_22050hz/', '')
        sharedkey = Path(path).parent.stem + '_' + Path(path).stem.replace(
            '_mel', '').split('_sample_')[0]
    elif dataset_name.lower() == 'caps':
        sharedkey = Path(path).stem.replace('_mel', '').split('_sample_')[0]
    else:
        raise NotImplementedError
    return sharedkey
