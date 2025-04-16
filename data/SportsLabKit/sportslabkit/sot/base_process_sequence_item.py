def process_sequence_item(self, sequence: Any):
    is_batched = isinstance(sequence, np.ndarray) and len(sequence.shape) == 4
    if is_batched:
        updated_states = self.update(sequence)
    else:
        updated_states = [self.update(sequence)]
    for updated_state in updated_states:
        self.check_updated_state(updated_state)
        self.update_tracklet_observations(updated_state)
        self.frame_count += 1
