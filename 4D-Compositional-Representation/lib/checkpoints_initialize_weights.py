def initialize_weights(self):
    """ Initializes the model weights from another model file.
        """
    print('Intializing weights from model %s' % self.initialize_from)
    filename_in = os.path.join(self.initialize_from, self.
        initialization_file_name)
    model_state_dict = self.module_dict.get('model').state_dict()
    model_dict = self.module_dict.get('model').state_dict()
    model_keys = set([k for k, v in model_dict.items()])
    init_model_dict = torch.load(filename_in)['model']
    init_model_k = set([k for k, v in init_model_dict.items()])
    for k in model_keys:
        if k in init_model_k and model_state_dict[k].shape == init_model_dict[k
            ].shape:
            model_state_dict[k] = init_model_dict[k]
    self.module_dict.get('model').load_state_dict(model_state_dict)
