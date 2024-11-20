@add_start_docstrings_to_model_forward(FLAUBERT_INPUTS_DOCSTRING)
@add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint=
    'flaubert/flaubert_base_cased', output_type=BaseModelOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, langs=None,
    token_type_ids=None, position_ids=None, lengths=None, cache=None,
    head_mask=None, inputs_embeds=None, output_attentions=None,
    output_hidden_states=None, return_dict=None):
    output_attentions = (output_attentions if output_attentions is not None
         else self.config.output_attentions)
    output_hidden_states = (output_hidden_states if output_hidden_states is not
        None else self.config.output_hidden_states)
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if input_ids is not None:
        bs, slen = input_ids.size()
    else:
        bs, slen = inputs_embeds.size()[:-1]
    device = (input_ids.device if input_ids is not None else inputs_embeds.
        device)
    if lengths is None:
        if input_ids is not None:
            lengths = (input_ids != self.pad_index).sum(dim=1).long()
        else:
            lengths = torch.tensor([slen] * bs, device=device)
    assert lengths.size(0) == bs
    assert lengths.max().item() <= slen
    mask, attn_mask = get_masks(slen, lengths, self.causal, padding_mask=
        attention_mask)
    if position_ids is None:
        position_ids = torch.arange(slen, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand((bs, slen))
    else:
        assert position_ids.size() == (bs, slen)
    if langs is not None:
        assert langs.size() == (bs, slen)
    head_mask = self.get_head_mask(head_mask, self.config.n_layers)
    if cache is not None and input_ids is not None:
        _slen = slen - cache['slen']
        input_ids = input_ids[:, -_slen:]
        position_ids = position_ids[:, -_slen:]
        if langs is not None:
            langs = langs[:, -_slen:]
        mask = mask[:, -_slen:]
        attn_mask = attn_mask[:, -_slen:]
    if inputs_embeds is None:
        inputs_embeds = self.embeddings(input_ids)
    tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(
        inputs_embeds)
    if langs is not None and self.use_lang_emb and self.config.n_langs > 1:
        tensor = tensor + self.lang_embeddings(langs)
    if token_type_ids is not None:
        tensor = tensor + self.embeddings(token_type_ids)
    tensor = self.layer_norm_emb(tensor)
    tensor = F.dropout(tensor, p=self.dropout, training=self.training)
    tensor *= mask.unsqueeze(-1).to(tensor.dtype)
    hidden_states = () if output_hidden_states else None
    attentions = () if output_attentions else None
    for i in range(self.n_layers):
        dropout_probability = random.uniform(0, 1)
        if self.training and dropout_probability < self.layerdrop:
            continue
        if output_hidden_states:
            hidden_states = hidden_states + (tensor,)
        if not self.pre_norm:
            attn_outputs = self.attentions[i](tensor, attn_mask, cache=
                cache, head_mask=head_mask[i], output_attentions=
                output_attentions)
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)
        else:
            tensor_normalized = self.layer_norm1[i](tensor)
            attn_outputs = self.attentions[i](tensor_normalized, attn_mask,
                cache=cache, head_mask=head_mask[i])
            attn = attn_outputs[0]
            if output_attentions:
                attentions = attentions + (attn_outputs[1],)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            tensor = tensor + attn
        if not self.pre_norm:
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)
        else:
            tensor_normalized = self.layer_norm2[i](tensor)
            tensor = tensor + self.ffns[i](tensor_normalized)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
    if output_hidden_states:
        hidden_states = hidden_states + (tensor,)
    if cache is not None:
        cache['slen'] += tensor.size(1)
    if not return_dict:
        return tuple(v for v in [tensor, hidden_states, attentions] if v is not
            None)
    return BaseModelOutput(last_hidden_state=tensor, hidden_states=
        hidden_states, attentions=attentions)
