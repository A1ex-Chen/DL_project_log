def compute_all_stream_relative_buckets(num_buckets, max_distance, position_ids
    ):
    """
    This function computes both main and predict relative position buckets. For more detail, see paper.
    """
    main_stream_relative_positions = position_ids.unsqueeze(1).repeat(1,
        position_ids.size(-1), 1)
    main_stream_relative_positions = (main_stream_relative_positions -
        position_ids.unsqueeze(-1))
    predicting_stream_relative_positions = torch.cat((position_ids - 1,
        position_ids), dim=-1).unsqueeze(1)
    predicting_stream_relative_positions = (
        predicting_stream_relative_positions.repeat(1, position_ids.size(-1
        ), 1))
    predicting_stream_relative_positions = (
        predicting_stream_relative_positions - position_ids.unsqueeze(-1))
    main_relative_position_buckets = compute_relative_buckets(num_buckets,
        max_distance, main_stream_relative_positions, is_bidirectional=False)
    predict_relative_position_buckets = compute_relative_buckets(num_buckets,
        max_distance, predicting_stream_relative_positions,
        is_bidirectional=False)
    return main_relative_position_buckets, predict_relative_position_buckets
