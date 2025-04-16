@add_start_docstrings_to_model_forward(ENCODER_DECODER_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=
    _CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=
    None, decoder_attention_mask=None, encoder_outputs=None,
    past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None,
    labels=None, use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None, **kwargs):
    """
        Returns:

        Examples::

            >>> from transformers import EncoderDecoderModel, BertTokenizer
            >>> import torch

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints

            >>> # forward
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)

            >>> # training
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
            >>> loss, logits = outputs.loss, outputs.logits

            >>> # save and load from pretrained
            >>> model.save_pretrained("bert2bert")
            >>> model = EncoderDecoderModel.from_pretrained("bert2bert")

            >>> # generation
            >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)

        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    kwargs_encoder = {argument: value for argument, value in kwargs.items() if
        not argument.startswith('decoder_')}
    kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in
        kwargs.items() if argument.startswith('decoder_')}
    if encoder_outputs is None:
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=
            attention_mask, inputs_embeds=inputs_embeds, output_attentions=
            output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, **kwargs_encoder)
    encoder_hidden_states = encoder_outputs[0]
    decoder_outputs = self.decoder(input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask, encoder_hidden_states=
        encoder_hidden_states, encoder_attention_mask=attention_mask,
        inputs_embeds=decoder_inputs_embeds, labels=labels,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict, **kwargs_decoder)
    if not return_dict:
        return decoder_outputs + encoder_outputs
    return Seq2SeqLMOutput(loss=decoder_outputs.loss, logits=
        decoder_outputs.logits, past_key_values=None, decoder_hidden_states
        =decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.
        attentions, cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions)
