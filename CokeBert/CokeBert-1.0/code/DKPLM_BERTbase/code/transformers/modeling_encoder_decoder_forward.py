def forward(self, encoder_input_ids, decoder_input_ids, **kwargs):
    """ The forward pass on a seq2eq depends what we are performing:

        - During training we perform one forward pass through both the encoder
          and decoder;
        - During prediction, we perform one forward pass through the encoder,
          and then perform several forward passes with the encoder's hidden
          state through the decoder to decode a full sequence.

        Therefore, we skip the forward pass on the encoder if an argument named
        `encoder_hidden_state` is passed to this function.

        Params:
            encoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of encoder input sequence tokens in the vocabulary.
            decoder_input_ids: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``
                Indices of decoder input sequence tokens in the vocabulary.
            kwargs: (`optional`) Remaining dictionary of keyword arguments.
        """
    kwargs_common = {argument: value for argument, value in kwargs.items() if
        not argument.startswith('encoder_') and not argument.startswith(
        'decoder_')}
    kwargs_decoder = kwargs_common.copy()
    kwargs_encoder = kwargs_common.copy()
    kwargs_encoder.update({argument[len('encoder_'):]: value for argument,
        value in kwargs.items() if argument.startswith('encoder_')})
    kwargs_decoder.update({argument[len('decoder_'):]: value for argument,
        value in kwargs.items() if argument.startswith('decoder_')})
    encoder_hidden_states = kwargs_encoder.pop('hidden_states', None)
    if encoder_hidden_states is None:
        encoder_outputs = self.encoder(encoder_input_ids, **kwargs_encoder)
        encoder_hidden_states = encoder_outputs[0]
    else:
        encoder_outputs = ()
    kwargs_decoder['encoder_hidden_states'] = encoder_hidden_states
    kwargs_decoder['encoder_attention_mask'] = kwargs_encoder.get(
        'attention_mask', None)
    decoder_outputs = self.decoder(decoder_input_ids, **kwargs_decoder)
    return decoder_outputs + encoder_outputs
