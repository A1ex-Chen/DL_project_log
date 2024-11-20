def _cleanup_checkpoints(self, trim=0):
    trim = min(len(self.checkpoint_files), trim)
    delete_index = self.max_history - trim
    if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
        return
    to_delete = self.checkpoint_files[delete_index:]
    for d in to_delete:
        try:
            LOGGER.debug('Cleaning checkpoint: %s', d)
            os.remove(d[0])
        except Exception as e:
            LOGGER.error("Exception '%s' while deleting checkpoint", e)
    self.checkpoint_files = self.checkpoint_files[:delete_index]
