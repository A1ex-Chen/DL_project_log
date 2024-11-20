def _rotate_checkpoints(self, use_mtime=False) ->None:
    if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
        return
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
    if len(checkpoints_sorted) <= self.args.save_total_limit:
        return
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self
        .args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:
        number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(
            'Deleting older checkpoint [{}] due to args.save_total_limit'.
            format(checkpoint))
        shutil.rmtree(checkpoint)
