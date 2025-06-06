def forward(self, input_ids: Optional[torch.Tensor]=None, attention_mask:
    Optional[torch.Tensor]=None, token_type_ids: Optional[torch.Tensor]=
    None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[
    torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=None,
    encoder_hidden_states: Optional[torch.Tensor]=None,
    encoder_attention_mask: Optional[torch.Tensor]=None, output_attentions:
    Optional[bool]=None, return_dict: Optional[bool]=None,
    output_hidden_states: Optional[bool]=None):
    """ """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    outputs = self.base_model(input_ids=input_ids, attention_mask=
        attention_mask, token_type_ids=token_type_ids, position_ids=
        position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states, encoder_attention_mask
        =encoder_attention_mask, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict)
    projection_state = self.transformation(outputs.last_hidden_state)
    return TransformationModelOutput(projection_state=projection_state,
        last_hidden_state=outputs.last_hidden_state, hidden_states=outputs.
        hidden_states, attentions=outputs.attentions)
