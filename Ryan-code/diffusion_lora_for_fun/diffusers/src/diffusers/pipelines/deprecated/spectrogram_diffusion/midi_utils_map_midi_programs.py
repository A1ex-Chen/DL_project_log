def map_midi_programs(feature, codec: Codec, granularity_type: str='full',
    feature_key: str='inputs') ->Mapping[str, Any]:
    """Apply MIDI program map to token sequences."""
    granularity = PROGRAM_GRANULARITIES[granularity_type]
    feature[feature_key] = granularity.tokens_map_fn(feature[feature_key],
        codec)
    return feature
