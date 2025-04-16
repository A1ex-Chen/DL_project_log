def __init__(self, dataset_folder, fields, split=None, categories=None,
    no_except=True, transform=None):
    """ Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        """
    self.dataset_folder = dataset_folder
    self.fields = fields
    self.no_except = no_except
    self.transform = transform
    if categories is None:
        categories = os.listdir(dataset_folder)
        categories = [c for c in categories if os.path.isdir(os.path.join(
            dataset_folder, c))]
    metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            self.metadata = yaml.load(f)
    else:
        self.metadata = {c: {'id': c, 'name': 'n/a'} for c in categories}
    for c_idx, c in enumerate(categories):
        self.metadata[c]['idx'] = c_idx
    self.models = []
    for c_idx, c in enumerate(categories):
        subpath = os.path.join(dataset_folder, c)
        if not os.path.isdir(subpath):
            logger.warning('Category %s does not exist in dataset.' % c)
        split_file = os.path.join(subpath, split + '.lst')
        with open(split_file, 'r') as f:
            models_c = f.read().split('\n')
        self.models += [{'category': c, 'model': m} for m in models_c]
