def _get_save_path(self):
    folder = Path(self.save_folder)
    filename = '{}-sample_size{}-seed{}.pkl'.format(self.code(), self.
        sample_size, self.seed)
    return folder.joinpath(filename)
