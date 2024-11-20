def get_checkpoint_file(self):
    save_file = os.path.join(self.save_dir, 'last_checkpoint' + self.postfix)
    try:
        self._last_checkpoints = self._load_last_checkpoints(save_file)
        last_saved = self._last_checkpoints[-1]
    except (IOError, IndexError):
        last_saved = ''
    return last_saved
