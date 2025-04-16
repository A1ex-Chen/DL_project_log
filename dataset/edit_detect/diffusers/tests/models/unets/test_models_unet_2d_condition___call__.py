def __call__(self, attn, hidden_states, encoder_hidden_states=None,
    attention_mask=None, number=None):
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = attn.prepare_attention_mask(attention_mask,
        sequence_length, batch_size)
    query = attn.to_q(hidden_states)
    encoder_hidden_states = (encoder_hidden_states if encoder_hidden_states
         is not None else hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    query = attn.head_to_batch_dim(query)
    key = attn.head_to_batch_dim(key)
    value = attn.head_to_batch_dim(value)
    attention_probs = attn.get_attention_scores(query, key, attention_mask)
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = attn.batch_to_head_dim(hidden_states)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    hidden_states += self.weight
    self.is_run = True
    self.counter += 1
    self.number = number
    return hidden_states
