def process_sequence_item(self, sequence: Any):
    self.frame_count += 1
    is_batched = isinstance(sequence, np.ndarray) and len(sequence.shape) == 4
    tracklets = self.alive_tracklets
    if is_batched:
        raise NotImplementedError('Batched tracking is not yet supported')
    assigned_tracklets, new_tracklets, unassigned_tracklets = self.update(
        sequence, tracklets)
    assigned_tracklets = self.reset_staleness(assigned_tracklets)
    unassigned_tracklets = self.increment_staleness(unassigned_tracklets)
    non_stale_tracklets, stale_tracklets = self.separate_stale_tracklets(
        unassigned_tracklets)
    stale_tracklets = self.cleanup_tracklets(stale_tracklets)
    logger.debug(
        f'assigned: {len(assigned_tracklets)}, new: {len(new_tracklets)}, unassigned: {len(non_stale_tracklets)}, stale: {len(stale_tracklets)}'
        )
    self.alive_tracklets = (assigned_tracklets + new_tracklets +
        non_stale_tracklets)
    self.dead_tracklets += stale_tracklets
