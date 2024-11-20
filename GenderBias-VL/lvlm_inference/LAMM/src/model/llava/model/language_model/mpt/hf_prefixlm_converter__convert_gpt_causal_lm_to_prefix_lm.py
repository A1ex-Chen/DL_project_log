def _convert_gpt_causal_lm_to_prefix_lm(model: CAUSAL_GPT_TYPES
    ) ->CAUSAL_GPT_TYPES:
    """Converts a GPT-style Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `GPT2LMHeadModel`
        - `GPTNeoForCausalLM`
        - `GPTNeoXForCausalLM`
        - `GPTJForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model
    assert isinstance(model, _SUPPORTED_GPT_MODELS)
    assert model.config.add_cross_attention == False, 'Only supports GPT-style decoder-only models'

    def _get_attn_modules(model: CAUSAL_GPT_TYPES) ->List[torch.nn.Module]:
        """Helper that gets a list of the model's attention modules.

        Each module has a `bias` buffer used for causal masking. The Prefix LM
        conversion adds logic to dynamically manipulate these biases to support
        Prefix LM attention masking.
        """
        attn_modules = []
        if isinstance(model, GPTNeoXForCausalLM):
            blocks = model.gpt_neox.layers
        else:
            blocks = model.transformer.h
        for block in blocks:
            if isinstance(model, GPTNeoForCausalLM):
                if block.attn.attention_type != 'global':
                    continue
                attn_module = block.attn.attention
            elif isinstance(model, GPTNeoXForCausalLM):
                attn_module = block.attention
            else:
                attn_module = block.attn
            attn_modules.append(attn_module)
        return attn_modules
    setattr(model, '_original_forward', getattr(model, 'forward'))
    setattr(model, '_original_generate', getattr(model, 'generate'))

    def forward(self: CAUSAL_GPT_TYPES, input_ids: Optional[torch.
        LongTensor]=None, past_key_values: Optional[Tuple[Tuple[torch.
        Tensor]]]=None, attention_mask: Optional[torch.FloatTensor]=None,
        bidirectional_mask: Optional[torch.Tensor]=None, token_type_ids:
        Optional[torch.LongTensor]=None, position_ids: Optional[torch.
        LongTensor]=None, head_mask: Optional[torch.FloatTensor]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[
        torch.LongTensor]=None, use_cache: Optional[bool]=None,
        output_attentions: Optional[bool]=None, output_hidden_states:
        Optional[bool]=None, return_dict: Optional[bool]=None):
        """Wraps original forward to enable PrefixLM attention."""

        def call_og_forward():
            if isinstance(self, GPTNeoXForCausalLM):
                return self._original_forward(input_ids=input_ids,
                    past_key_values=past_key_values, attention_mask=
                    attention_mask, head_mask=head_mask, inputs_embeds=
                    inputs_embeds, labels=labels, use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states, return_dict=
                    return_dict)
            else:
                return self._original_forward(input_ids=input_ids,
                    past_key_values=past_key_values, attention_mask=
                    attention_mask, token_type_ids=token_type_ids,
                    position_ids=position_ids, head_mask=head_mask,
                    inputs_embeds=inputs_embeds, labels=labels, use_cache=
                    use_cache, output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states, return_dict=
                    return_dict)
        if bidirectional_mask is None:
            return call_og_forward()
        assert isinstance(bidirectional_mask, torch.Tensor)
        attn_modules = _get_attn_modules(model)
        b, s = bidirectional_mask.shape
        max_length = attn_modules[0].bias.shape[-1]
        if s > max_length:
            raise ValueError(
                f'bidirectional_mask sequence length (={s}) exceeds the ' +
                f'max length allowed by the model ({max_length}).')
        assert s <= max_length
        if s < max_length:
            pad = torch.zeros((int(b), int(max_length - s)), dtype=
                bidirectional_mask.dtype, device=bidirectional_mask.device)
            bidirectional_mask = torch.cat([bidirectional_mask, pad], dim=1)
        bidirectional = bidirectional_mask.unsqueeze(1).unsqueeze(1)
        for attn_module in attn_modules:
            attn_module.bias.data = torch.logical_or(attn_module.bias.data,
                bidirectional)
        output = call_og_forward()
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[
                None, None]
        return output

    def generate(self: CAUSAL_GPT_TYPES, *args: tuple, **kwargs: Dict[str, Any]
        ):
        """Wraps original generate to enable PrefixLM attention."""
        attn_modules = _get_attn_modules(model)
        for attn_module in attn_modules:
            attn_module.bias.data[:] = 1
        output = self._original_generate(*args, **kwargs)
        for attn_module in attn_modules:
            attn_module.bias.data = torch.tril(attn_module.bias.data[0, 0])[
                None, None]
        return output
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'generate', MethodType(generate, model))
    setattr(model, '_prefix_lm_converted', True)
    return model
