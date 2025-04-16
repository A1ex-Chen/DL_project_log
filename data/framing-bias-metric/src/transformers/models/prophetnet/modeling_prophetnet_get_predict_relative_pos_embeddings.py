def get_predict_relative_pos_embeddings(self, hidden_states, attn_weights,
    position_ids, predict_relative_position_buckets):
    sequence_length, batch_size = hidden_states.shape[1:3]
    if predict_relative_position_buckets is None:
        key_sequence_length = attn_weights.shape[-1]
        assert position_ids[0][0
            ] == key_sequence_length - 1, '`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)'
        relative_positions = torch.arange(0, key_sequence_length).unsqueeze(0
            ).unsqueeze(0).repeat(batch_size, sequence_length, 1).to(
            position_ids.device)
        relative_positions = relative_positions - position_ids.unsqueeze(0
            ).repeat(batch_size, sequence_length, 1)
        predict_relative_position_buckets = compute_relative_buckets(self.
            num_buckets, self.relative_max_distance, relative_positions, False)
    hidden_states = hidden_states.transpose(1, 2)
    rel_pos_embeddings = self.relative_pos_embeddings(hidden_states).view(
        hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads))
    rel_pos_embeddings = rel_pos_embeddings.permute(0, 1, 4, 2, 3).reshape(
        self.ngram * batch_size * self.num_attn_heads, sequence_length, -1)
    predict_relative_position_buckets = (predict_relative_position_buckets.
        unsqueeze(0).repeat(self.ngram, 1, self.num_attn_heads, 1))
    rel_pos_embeddings = rel_pos_embeddings.reshape(-1, rel_pos_embeddings.
        size(-1))
    predict_relative_position_buckets = predict_relative_position_buckets.view(
        -1, predict_relative_position_buckets.size(-1)).long()
    predict_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=
        1, index=predict_relative_position_buckets).view(self.ngram, 
        batch_size * self.num_attn_heads, sequence_length, -1)
    return predict_relative_pos_embeddings
