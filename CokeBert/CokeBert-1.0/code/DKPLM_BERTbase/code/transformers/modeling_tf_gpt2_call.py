def call(self, inputs, past=None, attention_mask=None, token_type_ids=None,
    position_ids=None, head_mask=None, inputs_embeds=None, mc_token_ids=
    None, training=False):
    if isinstance(inputs, (tuple, list)):
        input_ids = inputs[0]
        past = inputs[1] if len(inputs) > 1 else past
        attention_mask = inputs[2] if len(inputs) > 2 else attention_mask
        token_type_ids = inputs[3] if len(inputs) > 3 else token_type_ids
        position_ids = inputs[4] if len(inputs) > 4 else position_ids
        head_mask = inputs[5] if len(inputs) > 5 else head_mask
        inputs_embeds = inputs[6] if len(inputs) > 6 else inputs_embeds
        mc_token_ids = inputs[7] if len(inputs) > 7 else mc_token_ids
        assert len(inputs) <= 8, 'Too many inputs.'
    elif isinstance(inputs, dict):
        input_ids = inputs.get('input_ids')
        past = inputs.get('past', past)
        attention_mask = inputs.get('attention_mask', attention_mask)
        token_type_ids = inputs.get('token_type_ids', token_type_ids)
        position_ids = inputs.get('position_ids', position_ids)
        head_mask = inputs.get('head_mask', head_mask)
        inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
        mc_token_ids = inputs.get('mc_token_ids', mc_token_ids)
        assert len(inputs) <= 8, 'Too many inputs.'
    else:
        input_ids = inputs
    if input_ids is not None:
        input_shapes = shape_list(input_ids)
    else:
        input_shapes = shape_list(inputs_embeds)[:-1]
    seq_length = input_shapes[-1]
    flat_input_ids = tf.reshape(input_ids, (-1, seq_length)
        ) if input_ids is not None else None
    flat_attention_mask = tf.reshape(attention_mask, (-1, seq_length)
        ) if attention_mask is not None else None
    flat_token_type_ids = tf.reshape(token_type_ids, (-1, seq_length)
        ) if token_type_ids is not None else None
    flat_position_ids = tf.reshape(position_ids, (-1, seq_length)
        ) if position_ids is not None else None
    flat_inputs = [flat_input_ids, past, flat_attention_mask,
        flat_token_type_ids, flat_position_ids, head_mask, inputs_embeds]
    transformer_outputs = self.transformer(flat_inputs, training=training)
    hidden_states = transformer_outputs[0]
    hidden_states = tf.reshape(hidden_states, input_shapes + shape_list(
        hidden_states)[-1:])
    lm_logits = self.transformer.wte(hidden_states, mode='linear')
    mc_logits = self.multiple_choice_head([hidden_states, mc_token_ids],
        training=training)
    mc_logits = tf.squeeze(mc_logits, axis=-1)
    outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
    return outputs
