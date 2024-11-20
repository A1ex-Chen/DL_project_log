def __init__(self, dataset_folder, fields, mode, split=None, category=
    'D-FAUST', length_sequence=17, n_files_per_sequence=-1, offset_sequence
    =0, ex_folder_name='pcl_seq', specific_model=None):
    """ Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            category (str): category of data
            split (str): which split is used
        """
    self.dataset_folder = dataset_folder
    self.fields = fields
    self.mode = mode
    self.length_sequence = length_sequence
    self.n_files_per_sequence = n_files_per_sequence
    self.offset_sequence = offset_sequence
    self.ex_folder_name = ex_folder_name
    if mode == 'train':
        with open(os.path.join(self.dataset_folder, 'train_human_ids.lst'), 'r'
            ) as f:
            self.hid = f.read().split('\n')
    if specific_model is not None:
        self.models = [{'category': category, 'model': specific_model['seq'
            ], 'start_idx': specific_model['start_idx']}]
    else:
        self.models = []
        subpath = os.path.join(self.dataset_folder, 'test', category)
        if split is not None and os.path.exists(os.path.join(subpath, split +
            '.lst')):
            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
        else:
            models_c = [f for f in os.listdir(subpath) if os.path.isdir(os.
                path.join(subpath, f))]
        models_c = list(filter(lambda x: len(x) > 0, models_c))
        models_len = self.get_models_seq_len(subpath, models_c)
        models_c, start_idx = self.subdivide_into_sequences(models_c,
            models_len)
        self.models += [{'category': category, 'model': m, 'start_idx':
            start_idx[i]} for i, m in enumerate(models_c)]
