def run_length_encode_shifts(features: MutableMapping[str, Any]) ->Mapping[
    str, Any]:
    """Combine leading/interior shifts, trim trailing shifts.

        Args:
          features: Dict of features to process.

        Returns:
          A dict of features.
        """
    events = features[feature_key]
    shift_steps = 0
    total_shift_steps = 0
    output = np.array([], dtype=np.int32)
    current_state = np.zeros(len(state_change_event_ranges), dtype=np.int32)
    for event in events:
        if codec.is_shift_event_index(event):
            shift_steps += 1
            total_shift_steps += 1
        else:
            is_redundant = False
            for i, (min_index, max_index) in enumerate(
                state_change_event_ranges):
                if min_index <= event and event <= max_index:
                    if current_state[i] == event:
                        is_redundant = True
                    current_state[i] = event
            if is_redundant:
                continue
            if shift_steps > 0:
                shift_steps = total_shift_steps
                while shift_steps > 0:
                    output_steps = np.minimum(codec.max_shift_steps,
                        shift_steps)
                    output = np.concatenate([output, [output_steps]], axis=0)
                    shift_steps -= output_steps
            output = np.concatenate([output, [event]], axis=0)
    features[feature_key] = output
    return features
