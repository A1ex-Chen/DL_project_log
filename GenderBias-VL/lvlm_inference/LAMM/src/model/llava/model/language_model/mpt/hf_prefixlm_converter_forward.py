def forward(self: OPTForCausalLM, input_ids: Optional[torch.LongTensor]=
    None, attention_mask: Optional[torch.Tensor]=None, bidirectional_mask:
    Optional[torch.ByteTensor]=None, head_mask: Optional[torch.Tensor]=None,
    past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds:
    Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=
    None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]
    =None, output_hidden_states: Optional[bool]=None, return_dict: Optional
    [bool]=None):

    def call_og_forward():
        return self._original_forward(input_ids=input_ids, attention_mask=
            attention_mask, head_mask=head_mask, past_key_values=
            past_key_values, inputs_embeds=inputs_embeds, labels=labels,
            use_cache=use_cache, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict)
    if bidirectional_mask is None:
        return call_og_forward()
    self.model.decoder.bidirectional_mask = bidirectional_mask
    try:
        outputs = call_og_forward()
    except:
        self.model.decoder.bidirectional_mask = None
        raise
    self.model.decoder.bidirectional_mask = None
    return outputs
