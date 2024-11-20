def save(self, filename, **kwargs):
    """ Saves the current module dictionary.

        Args:
            filename (str): name of output file
        """
    if not os.path.isabs(filename):
        filename = os.path.join(self.checkpoint_dir, filename)
    outdict = kwargs
    for k, v in self.module_dict.items():
        outdict[k] = v.state_dict()
    torch.save(outdict, filename)
