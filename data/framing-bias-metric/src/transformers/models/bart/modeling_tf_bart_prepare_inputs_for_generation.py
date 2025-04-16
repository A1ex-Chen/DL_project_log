def prepare_inputs_for_generation(self, decoder_input_ids, past,
    attention_mask, use_cache=True, **kwargs) ->Dict:
    assert past is not None and len(past) in {1, 2
        }, f'past has to be an iterable of length 1,2 got {past}'
    if len(past) == 1:
        assert isinstance(past[0], tf.Tensor)
        encoder_outputs = TFBaseModelOutput(last_hidden_state=past[0])
        decoder_cached_states = None
    else:
        assert len(past) == 2
        encoder_outputs, decoder_cached_states = past
        if isinstance(encoder_outputs, tuple):
            assert isinstance(encoder_outputs[0], tf.Tensor)
            encoder_outputs = TFBaseModelOutput(last_hidden_state=
                encoder_outputs[0])
        elif isinstance(encoder_outputs, tf.Tensor):
            encoder_outputs = TFBaseModelOutput(last_hidden_state=
                encoder_outputs)
        assert decoder_cached_states, f'decoder cached states must be truthy. got {decoder_cached_states} from the 2nd element of past'
    assert isinstance(encoder_outputs, TFBaseModelOutput
        ), f'encoder_outputs should be a TFBaseModelOutput, Instead got {type(encoder_outputs)}.'
    return {'input_ids': None, 'encoder_outputs': encoder_outputs,
        'past_key_values': decoder_cached_states, 'decoder_input_ids':
        decoder_input_ids, 'attention_mask': attention_mask, 'use_cache':
        use_cache}
