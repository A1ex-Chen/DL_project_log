def extract_sequence_with_indices(features, state_events_end_token=None,
    feature_key='inputs'):
    """Extract target sequence corresponding to audio token segment."""
    features = features.copy()
    start_idx = features['event_start_indices'][0]
    end_idx = features['event_end_indices'][-1]
    features[feature_key] = features[feature_key][start_idx:end_idx]
    if state_events_end_token is not None:
        state_event_start_idx = features['state_event_indices'][0]
        state_event_end_idx = state_event_start_idx + 1
        while features['state_events'][state_event_end_idx - 1
            ] != state_events_end_token:
            state_event_end_idx += 1
        features[feature_key] = np.concatenate([features['state_events'][
            state_event_start_idx:state_event_end_idx], features[
            feature_key]], axis=0)
    return features
