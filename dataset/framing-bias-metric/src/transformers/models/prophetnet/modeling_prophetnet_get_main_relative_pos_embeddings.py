def get_main_relative_pos_embeddings(self, hidden_states, attn_weights,
    position_ids, main_relative_position_buckets):
    if main_relative_position_buckets is None:
        batch_size, sequence_length = hidden_states.shape[:2]
        relative_positions = torch.arange(1, attn_weights.shape[-1] + 1
            ).unsqueeze(0).unsqueeze(0).repeat(batch_size, sequence_length, 1
            ).to(position_ids.device)
        relative_positions = relative_positions - position_ids.unsqueeze(0
            ).repeat(batch_size, sequence_length, 1)
        main_relative_position_buckets = compute_relative_buckets(self.
            num_buckets, self.relative_max_distance, relative_positions, False)
    hidden_states = hidden_states.transpose(0, 1)
    rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
    rel_pos_embeddings = rel_pos_embeddings.view(rel_pos_embeddings.shape[:
        2] + (self.num_buckets, self.num_attn_heads)).permute(0, 3, 1, 2)
    rel_pos_embeddings = rel_pos_embeddings.reshape(attn_weights.shape[:2] +
        (-1,))
    main_relative_position_buckets = main_relative_position_buckets.repeat(
        1, self.num_attn_heads, 1).view(-1, main_relative_position_buckets.
        shape[-1]).long()
    rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.
        size(-1))
    main_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1,
        index=main_relative_position_buckets).view(attn_weights.shape[:2] +
        (-1,))
    return main_relative_pos_embeddings
