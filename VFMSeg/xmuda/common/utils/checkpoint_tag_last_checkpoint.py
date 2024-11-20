def tag_last_checkpoint(self, last_filename):
    save_file = os.path.join(self.save_dir, 'last_checkpoint' + self.postfix)
    for path in self._last_checkpoints:
        if last_filename == path:
            self._last_checkpoints.remove(path)
    self._last_checkpoints.append(last_filename)
    self._delete_old_checkpoint()
    self._save_checkpoint_file(save_file)
