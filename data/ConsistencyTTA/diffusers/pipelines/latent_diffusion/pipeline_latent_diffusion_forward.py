def forward(self, input_ids=None, attention_mask=None, position_ids=None,
    head_mask=None, inputs_embeds=None, output_attentions=None,
    output_hidden_states=None, return_dict=None):
    outputs = self.model(input_ids, attention_mask=attention_mask,
        position_ids=position_ids, head_mask=head_mask, inputs_embeds=
        inputs_embeds, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict)
    return outputs
