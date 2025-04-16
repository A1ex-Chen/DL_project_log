def _feature_file(self, mode):
    return os.path.join(self.hparams.data_dir, 'cached_{}_{}_{}'.format(
        mode, list(filter(None, self.hparams.model_name_or_path.split('/'))
        ).pop(), str(self.hparams.max_seq_length)))
