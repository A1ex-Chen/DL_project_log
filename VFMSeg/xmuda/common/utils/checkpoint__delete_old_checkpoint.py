def _delete_old_checkpoint(self):
    if len(self._last_checkpoints) > self.max_to_keep:
        path = self._last_checkpoints.pop(0)
        try:
            os.remove(path)
        except Exception as e:
            logging.warning('Ignoring: %s', str(e))
