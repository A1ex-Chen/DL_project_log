def load_file(self, filename):
    """Loads a module dictionary from file.

        Args:
            filename (str): name of saved module dictionary
        """
    if not os.path.isabs(filename):
        filename = os.path.join(self.checkpoint_dir, filename)
    if os.path.exists(filename):
        print(filename)
        print('=> Loading checkpoint from local file...')
        state_dict = torch.load(filename)
        scalars = self.parse_state_dict(state_dict)
        return scalars
    else:
        if self.initialize_from is not None:
            self.initialize_weights()
        raise FileExistsError
