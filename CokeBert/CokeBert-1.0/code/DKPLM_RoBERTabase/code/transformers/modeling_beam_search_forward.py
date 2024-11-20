def forward(self, encoder_input_ids, **kwargs):
    kwargs_encoder = {argument[len('encoder_'):]: value for argument, value in
        kwargs.items() if argument.startswith('encoder_')}
    kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in
        kwargs.items() if argument.startswith('decoder_')}
    kwargs_common = {argument: value for argument, value in kwargs.items() if
        not (argument.startswith('encoder_') or argument.startswith(
        'decoder_'))}
    kwargs_decoder = dict(kwargs_common, **kwargs_decoder)
    kwargs_encoder = dict(kwargs_common, **kwargs_encoder)
    encoder_outputs = self.model.encoder.forward(encoder_input_ids,
        kwargs_encoder)
    kwargs_decoder['encoder_hidden_states'] = tile(encoder_outputs, self.
        beam_size, dim=0)
    self.growing_beam = torch.full((self.batch_size * self.beam_size, 1),
        self.start_token_id, dtype=torch.long)
    for step in range(self.max_length):
        decoder_input = self.growing_beam[:, -1]
        outputs = self.model.decoder(decoder_input, kwargs_decoder)
        log_probabilities = torch.nn.functional.log_softmax(outputs[1])
        surviving_beams_rows = self.step(log_probabilities)
        if self.is_done:
            break
        kwargs_decoder['encoder_hidden_states'] = kwargs_decoder[
            'encoder_hidden_states'].index_select(0, surviving_beams_rows)
    return self.results
