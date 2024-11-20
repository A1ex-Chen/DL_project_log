def forward(self, input_ids=None, attention_mask=None, langs=None,
    token_type_ids=None, position_ids=None, lengths=None, cache=None,
    head_mask=None, inputs_embeds=None, start_positions=None, end_positions
    =None, is_impossible=None, cls_index=None, p_mask=None):
    transformer_outputs = self.transformer(input_ids, attention_mask=
        attention_mask, langs=langs, token_type_ids=token_type_ids,
        position_ids=position_ids, lengths=lengths, cache=cache, head_mask=
        head_mask, inputs_embeds=inputs_embeds)
    output = transformer_outputs[0]
    outputs = self.qa_outputs(output, start_positions=start_positions,
        end_positions=end_positions, cls_index=cls_index, is_impossible=
        is_impossible, p_mask=p_mask)
    outputs = outputs + transformer_outputs[1:]
    return outputs
