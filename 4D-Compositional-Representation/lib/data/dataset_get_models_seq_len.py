def get_models_seq_len(self, subpath, models):
    """ Returns the sequence length of a specific model.

        This is a little "hacky" as we assume the existence of the folder
        self.ex_folder_name. However, in our case this is always given.

        Args:
            subpath (str): subpath of model category
            models (list): list of model names
        """
    ex_folder_name = self.ex_folder_name
    models_seq_len = [len(os.listdir(os.path.join(subpath, m,
        ex_folder_name))) for m in models]
    return models_seq_len
