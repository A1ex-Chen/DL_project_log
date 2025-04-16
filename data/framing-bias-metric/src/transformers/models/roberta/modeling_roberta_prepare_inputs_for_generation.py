def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **
    model_kwargs):
    input_shape = input_ids.shape
    if attention_mask is None:
        attention_mask = input_ids.new_ones(input_shape)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}
