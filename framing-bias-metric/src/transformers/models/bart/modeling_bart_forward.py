@add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
    decoder_attention_mask=None, encoder_outputs=None, past_key_values=None,
    labels=None, use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None, extra_task=None,
    extra_task_input_ids=None, extra_task_attention_mask=None,
    extra_task_label=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:

        Conditional generation example::

            >>> # Mask filling only works for bart-large
            >>> from transformers import BartTokenizer, BartForConditionalGeneration
            >>> tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            >>> TXT = "My friends are <mask> but they eat too many carbs."

            >>> model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
            >>> input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']
            >>> logits = model(input_ids).logits

            >>> masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            >>> probs = logits[0, masked_index].softmax(dim=0)
            >>> values, predictions = probs.topk(5)

            >>> tokenizer.decode(predictions).split()
            >>> # ['good', 'great', 'all', 'really', 'very']
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    if labels is not None:
        use_cache = False
        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(labels, self.config.
                pad_token_id)
    outputs = self.model(input_ids, attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids, encoder_outputs=
        encoder_outputs, decoder_attention_mask=decoder_attention_mask,
        past_key_values=past_key_values, use_cache=use_cache,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.
        final_logits_bias)
    extra_task_loss = None
    if extra_task in ['webis', 'propaganda', 'clickbait', 'lr_vs_roundup']:
        extra_task_encoder_outputs = self.model(input_ids=
            extra_task_input_ids, attention_mask=extra_task_attention_mask,
            return_dict=return_dict)
        x = extra_task_encoder_outputs[0]
        eos_mask = extra_task_input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError(
                'All examples must have the same number of <eos> tokens.')
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1)
            )[:, -1, :]
        extra_task_logits = self.classification_head(sentence_representation)
        loss_fct = CrossEntropyLoss()
        extra_task_loss = loss_fct(extra_task_logits.view(-1, self.
            extra_task_num_labels), extra_task_label.view(-1))
    if not return_dict:
        output = (lm_logits,) + outputs[1:]
        return (extra_task_loss,
            ) + output if masked_lm_loss is not None else output
    return Seq2SeqLMOutput(loss=extra_task_loss, logits=lm_logits,
        past_key_values=outputs.past_key_values, decoder_hidden_states=
        outputs.decoder_hidden_states, decoder_attentions=outputs.
        decoder_attentions, cross_attentions=outputs.cross_attentions,
        encoder_last_hidden_state=outputs.encoder_last_hidden_state,
        encoder_hidden_states=outputs.encoder_hidden_states,
        encoder_attentions=outputs.encoder_attentions)
