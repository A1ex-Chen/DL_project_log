def prepare_inputs_for_generation(self, input_ids, query_embeds=None,
    past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
    if past_key_values:
        input_ids = input_ids[:, -1:]
    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -1].unsqueeze(-1)
            query_embeds = None
    per_input_len: torch.Tensor = attention_mask.sum(1).tolist()
    batch_size = input_ids.shape[0]
    if inputs_embeds is not None and self.infer_state is None:
        C = inputs_embeds.shape[-1]
        attention_mask = attention_mask.view(batch_size, -1, 1).to(dtype=bool)
        inputs_embeds = torch.masked_select(inputs_embeds, attention_mask)
        model_inputs = {'inputs_embeds': inputs_embeds.view(-1, C)}
    if self.infer_state == None:
        self.init_infer_state(batch_size=batch_size, total_token_num=sum(
            per_input_len), max_input_len=max(per_input_len))
        self.init_buffer(per_input_len=per_input_len, max_input_len=max(
            per_input_len), max_output_len=400)
    else:
        self.update_infer_state()
        self.update_buffer()
        model_inputs = {'input_ids': input_ids[:, -1]}
    model_inputs.update({'position_ids': position_ids, 'query_embeds':
        query_embeds, 'past_key_values': past_key_values, 'use_cache':
        kwargs.get('use_cache'), 'attention_mask': attention_mask})
    return model_inputs
