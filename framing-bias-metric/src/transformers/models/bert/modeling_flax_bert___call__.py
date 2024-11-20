@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format(
    'batch_size, sequence_length'))
def __call__(self, input_ids, attention_mask=None, token_type_ids=None,
    position_ids=None):
    if token_type_ids is None:
        token_type_ids = jnp.ones_like(input_ids)
    if position_ids is None:
        position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
    if attention_mask is None:
        attention_mask = jnp.ones_like(input_ids)
    return self.model.apply({'params': self.params}, jnp.array(input_ids,
        dtype='i4'), jnp.array(attention_mask, dtype='i4'), jnp.array(
        token_type_ids, dtype='i4'), jnp.array(position_ids, dtype='i4'))
