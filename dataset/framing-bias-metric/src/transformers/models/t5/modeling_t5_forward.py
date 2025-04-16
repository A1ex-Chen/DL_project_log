@add_start_docstrings_to_model_forward(T5_ENCODER_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=BaseModelOutput, config_class=
    _CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, head_mask=None,
    inputs_embeds=None, output_attentions=None, output_hidden_states=None,
    return_dict=None):
    """
        Returns:

        Example::

            >>> from transformers import T5Tokenizer, T5EncoderModel
            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5EncoderModel.from_pretrained('t5-small')
            >>> input_ids = tokenizer("Studies have been shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model(input_ids=input_ids)
            >>> last_hidden_states = outputs.last_hidden_state
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=
        attention_mask, inputs_embeds=inputs_embeds, head_mask=head_mask,
        output_attentions=output_attentions, output_hidden_states=
        output_hidden_states, return_dict=return_dict)
    return encoder_outputs
