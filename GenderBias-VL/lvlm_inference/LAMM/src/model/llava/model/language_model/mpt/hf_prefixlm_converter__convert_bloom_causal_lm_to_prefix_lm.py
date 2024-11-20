def _convert_bloom_causal_lm_to_prefix_lm(model: BloomForCausalLM
    ) ->BloomForCausalLM:
    """Converts a BLOOM Causal LM to a Prefix LM.

    Supported HuggingFace model classes:
        - `BloomForCausalLM`

    See `convert_hf_causal_lm_to_prefix_lm` for more details.
    """
    if hasattr(model, '_prefix_lm_converted'):
        return model
    assert isinstance(model, BloomForCausalLM)
    assert model.config.add_cross_attention == False, 'Only supports BLOOM decoder-only models'

    def _prepare_attn_mask(self: BloomModel, attention_mask: torch.Tensor,
        bidirectional_mask: Optional[torch.Tensor], input_shape: Tuple[int,
        int], past_key_values_length: int) ->torch.BoolTensor:
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape
        if src_length > 1:
            combined_attention_mask = _make_causal_mask_bloom(input_shape,
                device=device, past_key_values_length=past_key_values_length)
            if bidirectional_mask is not None:
                assert attention_mask.shape == bidirectional_mask.shape
                expanded_bidirectional_mask = _expand_mask_bloom(
                    bidirectional_mask, tgt_length=src_length)
                combined_attention_mask = torch.logical_and(
                    combined_attention_mask, expanded_bidirectional_mask)
        expanded_attn_mask = _expand_mask_bloom(attention_mask, tgt_length=
            src_length)
        combined_attention_mask = (expanded_attn_mask if 
            combined_attention_mask is None else expanded_attn_mask |
            combined_attention_mask)
        return combined_attention_mask

    def _build_alibi_tensor(self: BloomModel, batch_size: int, query_length:
        int, key_length: int, dtype: torch.dtype, device: torch.device
        ) ->torch.Tensor:
        num_heads = self.config.n_head
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        base = torch.tensor(2 ** -2 ** -(math.log2(closest_power_of_2) - 3),
            device=device, dtype=torch.float32)
        powers = torch.arange(1, 1 + closest_power_of_2, device=device,
            dtype=torch.int32)
        slopes = torch.pow(base, powers)
        if closest_power_of_2 != num_heads:
            extra_base = torch.tensor(2 ** -2 ** -(math.log2(2 *
                closest_power_of_2) - 3), device=device, dtype=torch.float32)
            num_remaining_heads = min(closest_power_of_2, num_heads -
                closest_power_of_2)
            extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2,
                device=device, dtype=torch.int32)
            slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)
                ], dim=0)
        qa = torch.arange(query_length, device=device, dtype=torch.int32).view(
            -1, 1)
        ka = torch.arange(key_length, device=device, dtype=torch.int32).view(
            1, -1)
        diffs = qa - ka + key_length - query_length
        diffs = -diffs.abs()
        alibi = slopes.view(1, num_heads, 1, 1) * diffs.view(1, 1,
            query_length, key_length)
        alibi = alibi.expand(batch_size, -1, -1, -1).reshape(-1,
            query_length, key_length)
        return alibi.to(dtype)
    KeyValueT = Tuple[torch.Tensor, torch.Tensor]

    def forward(self: BloomModel, input_ids: Optional[torch.LongTensor]=
        None, past_key_values: Optional[Tuple[KeyValueT, ...]]=None,
        attention_mask: Optional[torch.Tensor]=None, bidirectional_mask:
        Optional[torch.Tensor]=None, head_mask: Optional[torch.LongTensor]=
        None, inputs_embeds: Optional[torch.LongTensor]=None, use_cache:
        Optional[bool]=None, output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None, return_dict: Optional[
        bool]=None, **deprecated_arguments) ->Union[Tuple[torch.Tensor, ...
        ], BaseModelOutputWithPastAndCrossAttentions]:
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn(
                '`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. '
                 + 'You can safely ignore passing `position_ids`.',
                FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(
                f'Got unexpected arguments: {deprecated_arguments}')
        output_attentions = (output_attentions if output_attentions is not
            None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if 
            output_hidden_states is not None else self.config.
            output_hidden_states)
        use_cache = (use_cache if use_cache is not None else self.config.
            use_cache)
        return_dict = (return_dict if return_dict is not None else self.
            config.use_return_dict)
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                'You cannot specify both input_ids and inputs_embeds at the same time'
                )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                'You have to specify either input_ids or inputs_embeds')
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values[0] is not None:
            tmp = past_key_values[0][0]
            past_key_values_length = tmp.shape[2]
            seq_length_with_past = (seq_length_with_past +
                past_key_values_length)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past),
                device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)
        alibi = self._build_alibi_tensor(batch_size=batch_size,
            query_length=seq_length, key_length=seq_length_with_past, dtype
            =hidden_states.dtype, device=hidden_states.device)
        causal_mask = self._prepare_attn_mask(attention_mask,
            bidirectional_mask, input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length)
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                hst = hidden_states,
                all_hidden_states = all_hidden_states + hst
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        '`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...'
                        )
                    use_cache = False

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, use_cache=use_cache,
                            output_attentions=output_attentions)
                    return custom_forward
                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block), hidden_states, alibi,
                    causal_mask, head_mask[i])
            else:
                outputs = block(hidden_states, layer_past=layer_past,
                    attention_mask=causal_mask, head_mask=head_mask[i],
                    use_cache=use_cache, output_attentions=
                    output_attentions, alibi=alibi)
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                oa = outputs[2 if use_cache else 1],
                all_self_attentions = all_self_attentions + oa
        hidden_states = self.ln_f(hidden_states)
        if output_hidden_states:
            hst = hidden_states,
            all_hidden_states = all_hidden_states + hst
        if not return_dict:
            return tuple(v for v in [hidden_states, presents,
                all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=
            hidden_states, past_key_values=presents, hidden_states=
            all_hidden_states, attentions=all_self_attentions)
    setattr(model.transformer, '_prepare_attn_mask', MethodType(
        _prepare_attn_mask, model.transformer))
    setattr(model.transformer, '_build_alibi_tensor', MethodType(
        _build_alibi_tensor, model.transformer))
    setattr(model.transformer, 'forward', MethodType(forward, model.
        transformer))
    KeyValueT = Tuple[torch.Tensor, torch.Tensor]

    def forward(self: BloomForCausalLM, input_ids: Optional[torch.
        LongTensor]=None, past_key_values: Optional[Tuple[KeyValueT, ...]]=
        None, attention_mask: Optional[torch.Tensor]=None,
        bidirectional_mask: Optional[torch.Tensor]=None, head_mask:
        Optional[torch.Tensor]=None, inputs_embeds: Optional[torch.Tensor]=
        None, labels: Optional[torch.Tensor]=None, use_cache: Optional[bool
        ]=None, output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None, return_dict: Optional[
        bool]=None, **deprecated_arguments) ->Union[Tuple[torch.Tensor],
        CausalLMOutputWithCrossAttentions]:
        """Replacement forward method for BloomCausalLM."""
        if deprecated_arguments.pop('position_ids', False) is not False:
            warnings.warn(
                '`position_ids` have no functionality in BLOOM and will be removed '
                 +
                'in v5.0.0. You can safely ignore passing `position_ids`.',
                FutureWarning)
        if len(deprecated_arguments) > 0:
            raise ValueError(
                f'Got unexpected arguments: {deprecated_arguments}')
        return_dict = (return_dict if return_dict is not None else self.
            config.use_return_dict)
        transformer_outputs = self.transformer(input_ids, past_key_values=
            past_key_values, attention_mask=attention_mask,
            bidirectional_mask=bidirectional_mask, head_mask=head_mask,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=
            output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(batch_size * seq_length,
                vocab_size), shift_labels.view(batch_size * seq_length))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithCrossAttentions(loss=loss, logits=
            lm_logits, past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states, attentions=
            transformer_outputs.attentions)

    def prepare_inputs_for_generation(self: BloomForCausalLM, input_ids:
        torch.LongTensor, past: Optional[torch.Tensor]=None, attention_mask:
        Optional[torch.Tensor]=None, **kwargs) ->dict:
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            bidirectional_mask = None
            if past[0][0].shape[0] == input_ids.shape[0]:
                past = self._convert_to_bloom_cache(past)
        else:
            bidirectional_mask = torch.ones_like(input_ids)
        return {'input_ids': input_ids, 'past_key_values': past,
            'use_cache': True, 'attention_mask': attention_mask,
            'bidirectional_mask': bidirectional_mask}
    setattr(model, 'forward', MethodType(forward, model))
    setattr(model, 'prepare_inputs_for_generation', MethodType(
        prepare_inputs_for_generation, model))
    setattr(model, '_prefix_lm_converted', True)
    return model
