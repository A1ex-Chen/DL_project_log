def construct_dataset(clearml_info_string):
    """Load in a clearml dataset and fill the internal data_dict with its contents.
    """
    dataset_id = clearml_info_string.replace('clearml://', '')
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_root_path = Path(dataset.get_local_copy())
    yaml_filenames = list(glob.glob(str(dataset_root_path / '*.yaml')) +
        glob.glob(str(dataset_root_path / '*.yml')))
    if len(yaml_filenames) > 1:
        raise ValueError(
            'More than one yaml file was found in the dataset root, cannot determine which one contains the dataset definition this way.'
            )
    elif len(yaml_filenames) == 0:
        raise ValueError(
            'No yaml definition found in dataset root path, check that there is a correct yaml file inside the dataset root path.'
            )
    with open(yaml_filenames[0]) as f:
        dataset_definition = yaml.safe_load(f)
    assert set(dataset_definition.keys()).issuperset({'train', 'test',
        'val', 'nc', 'names'}
        ), "The right keys were not found in the yaml file, make sure it at least has the following keys: ('train', 'test', 'val', 'nc', 'names')"
    data_dict = dict()
    data_dict['train'] = str((dataset_root_path / dataset_definition[
        'train']).resolve()) if dataset_definition['train'] else None
    data_dict['test'] = str((dataset_root_path / dataset_definition['test']
        ).resolve()) if dataset_definition['test'] else None
    data_dict['val'] = str((dataset_root_path / dataset_definition['val']).
        resolve()) if dataset_definition['val'] else None
    data_dict['nc'] = dataset_definition['nc']
    data_dict['names'] = dataset_definition['names']
    return data_dict
