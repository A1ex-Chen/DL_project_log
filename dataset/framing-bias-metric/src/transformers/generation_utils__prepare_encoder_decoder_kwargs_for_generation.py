def _prepare_encoder_decoder_kwargs_for_generation(self, input_ids: torch.
    LongTensor, model_kwargs) ->Dict[str, Any]:
    encoder = self.get_encoder()
    encoder_kwargs = {argument: value for argument, value in model_kwargs.
        items() if not argument.startswith('decoder_')}
    model_kwargs['encoder_outputs']: ModelOutput = encoder(input_ids,
        return_dict=True, **encoder_kwargs)
    return model_kwargs
