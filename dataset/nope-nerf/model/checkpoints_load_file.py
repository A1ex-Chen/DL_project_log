def load_file(self, filename, device=None, load_model_only=False):
    """Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        """
    if not os.path.isabs(filename):
        filename = os.path.join(self.checkpoint_dir, filename)
    if os.path.exists(filename):
        print(filename)
        print('=> Loading checkpoint from local file...')
        if device is not None:
            state_dict = torch.load(filename, map_location=device)
        else:
            state_dict = torch.load(filename)
        if load_model_only:
            state_dict_model = {}
            state_dict_model['model'] = state_dict['model']
        else:
            state_dict_model = state_dict
        scalars = self.parse_state_dict(state_dict_model)
        return scalars
    else:
        raise FileExistsError
