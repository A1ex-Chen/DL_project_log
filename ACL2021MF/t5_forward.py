def forward(self, input_ids=None, attention_mask=None, encoder_img_mask=
    None, encoder_obj_feature=None, encoder_obj_box=None,
    encoder_relative_pos_index=None, decoder_copy_pos=None,
    decoder_concept_cls=None, decoder_input_ids=None,
    decoder_attention_mask=None, decoder_mention_flag=None,
    decoder_copy_mention_flag=None, decoder_cls_on_input=None,
    encoder_outputs=None, past_key_values=None, head_mask=None,
    inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=
    None, output_attentions=None, output_hidden_states=None, return_dict=
    None, decoder_history_input_ids=None, **kwargs):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you ", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
        """
    if 'lm_labels' in kwargs:
        warnings.warn(
            'The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.'
            , FutureWarning)
        labels = kwargs.pop('lm_labels')
    if 'decoder_past_key_value_states' in kwargs:
        warnings.warn(
            'The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.'
            , FutureWarning)
        past_key_values = kwargs.pop('decoder_past_key_value_states')
    if 'decoder_past_key_values' in kwargs:
        warnings.warn(
            'The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.'
            , FutureWarning)
        past_key_values = kwargs.pop('decoder_past_key_values')
    assert kwargs == {
        }, f'Unexpected keyword arguments: {list(kwargs.keys())}.'
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if encoder_outputs is None:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=
            attention_mask, encoder_img_mask=encoder_img_mask,
            encoder_obj_feature=encoder_obj_feature, encoder_obj_box=
            encoder_obj_box, encoder_relative_pos_index=
            encoder_relative_pos_index, inputs_embeds=inputs_embeds,
            head_mask=head_mask, output_attentions=output_attentions,
            output_hidden_states=output_hidden_states, return_dict=return_dict)
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs
            [0], hidden_states=encoder_outputs[1] if len(encoder_outputs) >
            1 else None, attentions=encoder_outputs[2] if len(
            encoder_outputs) > 2 else None)
    hidden_states = encoder_outputs[0]
    if (labels is not None and decoder_input_ids is None and 
        decoder_inputs_embeds is None):
        decoder_input_ids = self._shift_right(labels)
    if past_key_values is not None:
        assert labels is None, 'Decoder should not use cached key value states when training.'
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        if decoder_inputs_embeds is not None:
            decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]
    B = decoder_input_ids.size(0)
    repeat_mask = torch.zeros((B, self.config.vocab_size)).bool().to(
        decoder_input_ids.device)
    if not self.training:
        B_len = decoder_history_input_ids.size(1)
        decoder_history_input_ids = decoder_history_input_ids.detach().cpu(
            ).numpy().tolist()
        cur_decoder_mention_flag = decoder_mention_flag.clone().squeeze(1)
        d_cls = decoder_cls_on_input.detach().cpu().numpy().tolist()
        d_mf = cur_decoder_mention_flag.detach().cpu().tolist()
        batch_has_overlap = []
        batch_found_cls = []
        for i in range(B):
            available_cls = set()
            for cls_index, mf in zip(d_cls[i], d_mf[i]):
                if mf == 1 or mf == 2:
                    available_cls.add(cls_index)
            has_overlap = False
            leading_ch = set()
            for cls_ in available_cls:
                if self.copy_vocab.token_class[cls_][0] not in leading_ch:
                    leading_ch.add(self.copy_vocab.token_class[cls_][0])
                else:
                    has_overlap = True
            batch_has_overlap.append(has_overlap)
            has_repeat = False
            all_fgs = []
            min_len = {}
            for cls_index in available_cls:
                all_fgs += [(fg_index, cls_index) for _, fg_index in self.
                    copy_vocab.d_to_w_group[cls_index]]
                min_len[cls_index] = min([len(self.copy_vocab.token_fg_w[
                    fg_index]) for _, fg_index in self.copy_vocab.
                    d_to_w_group[cls_index]])
            all_fgs = sorted(all_fgs, key=lambda x: (min_len[x[1]], len(
                self.copy_vocab.token_fg_w[x[0]])), reverse=True)
            matched_position = []
            found_cls = set()
            for fg_index, cls_index in all_fgs:
                s1 = self.fg_str_dict[fg_index]
                fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
                for ch_idx, first_ch in enumerate(decoder_history_input_ids[i]
                    ):
                    if ch_idx in matched_position:
                        continue
                    if first_ch == fg_ch_list[0]:
                        s2 = '&'.join([str(f) for f in
                            decoder_history_input_ids[i][ch_idx:ch_idx +
                            len(fg_ch_list)]])
                        if s1 == s2:
                            if not has_overlap:
                                if cls_index not in found_cls:
                                    found_cls.add(cls_index)
                                    matched_position += [i for i in range(
                                        ch_idx, ch_idx + len(fg_ch_list))]
                                else:
                                    has_repeat = True
                            elif ch_idx + len(fg_ch_list
                                ) < B_len and decoder_history_input_ids[i][
                                ch_idx + len(fg_ch_list)
                                ] not in self.attachable_index:
                                if cls_index not in found_cls:
                                    found_cls.add(cls_index)
                                    matched_position += [i for i in range(
                                        ch_idx, ch_idx + len(fg_ch_list))]
                                else:
                                    has_repeat = True
                            elif ch_idx + len(fg_ch_list) == B_len:
                                matched_position += [i for i in range(
                                    ch_idx, ch_idx + len(fg_ch_list))]
                    if has_repeat:
                        break
                if has_repeat:
                    break
            batch_found_cls.append(found_cls)
            if self.local_config.decode_constrain == 'GBS':
                if has_repeat:
                    repeat_mask[i, :] = True
                else:
                    banned_first_word = set()
                    exclude_ban_word = set()
                    for cls_index in available_cls:
                        if cls_index not in found_cls:
                            for _, fg_index in self.copy_vocab.d_to_w_group[
                                cls_index]:
                                for ch in self.copy_vocab.token_fg_w[fg_index]:
                                    exclude_ban_word.add(ch)
                    for cls_index in found_cls:
                        for _, fg_index in self.copy_vocab.d_to_w_group[
                            cls_index]:
                            ban_ch = self.copy_vocab.token_fg_w[fg_index][0]
                            if ban_ch not in exclude_ban_word:
                                banned_first_word.add(ban_ch)
                    for wid in banned_first_word:
                        repeat_mask[i, wid] = True
        if self.local_config.use_mention_flag:
            if not self.local_config.static_mf:
                for i in range(B):
                    available_cls = set()
                    for cls_index, mf in zip(d_cls[i], d_mf[i]):
                        if mf == 1:
                            available_cls.add(cls_index)
                    has_overlap = batch_has_overlap[i]
                    for cls_index in available_cls:
                        state_number = 1
                        for _, fg_index in self.copy_vocab.d_to_w_group[
                            cls_index]:
                            s1 = self.fg_str_dict[fg_index]
                            fg_ch_list = self.copy_vocab.token_fg_w[fg_index]
                            for ch_idx, first_ch in enumerate(
                                decoder_history_input_ids[i]):
                                if first_ch == fg_ch_list[0]:
                                    s2 = '&'.join([str(f) for f in
                                        decoder_history_input_ids[i][ch_idx
                                        :ch_idx + len(fg_ch_list)]])
                                    if s1 == s2:
                                        if not has_overlap:
                                            state_number = 2
                                            break
                                        elif ch_idx + len(fg_ch_list
                                            ) < B_len and decoder_history_input_ids[
                                            i][ch_idx + len(fg_ch_list)
                                            ] not in self.attachable_index:
                                            state_number = 2
                                            break
                            if state_number == 2:
                                break
                        if state_number == 2:
                            cur_decoder_mention_flag[i][
                                decoder_cls_on_input[i] == cls_index
                                ] = state_number
            decoder_mention_flag = cur_decoder_mention_flag.unsqueeze(1)
    decoder_outputs = self.decoder(input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask, inputs_embeds=
        decoder_inputs_embeds, past_key_values=past_key_values,
        encoder_hidden_states=hidden_states, encoder_attention_mask=
        attention_mask, head_mask=head_mask, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict, mention_flag=
        decoder_mention_flag)
    sequence_output = decoder_outputs[0]
    sequence_output = sequence_output * self.model_dim ** -0.5
    lm_logits = self.lm_head(sequence_output)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return (loss,) + output if loss is not None else output
    return Seq2SeqLMOutputMF(loss=loss, logits=lm_logits, past_key_values=
        decoder_outputs.past_key_values, decoder_hidden_states=
        decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.
        attentions, cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
        decoder_mention_flags=decoder_mention_flag, repeat_mask=repeat_mask)
