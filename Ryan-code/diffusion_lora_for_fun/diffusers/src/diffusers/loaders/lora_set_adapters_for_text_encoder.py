def set_adapters_for_text_encoder(self, adapter_names: Union[List[str], str
    ], text_encoder: Optional['PreTrainedModel']=None, text_encoder_weights:
    Optional[Union[float, List[float], List[None]]]=None):
    """
        Sets the adapter layers for the text encoder.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            text_encoder (`torch.nn.Module`, *optional*):
                The text encoder module to set the adapter layers for. If `None`, it will try to get the `text_encoder`
                attribute.
            text_encoder_weights (`List[float]`, *optional*):
                The weights to use for the text encoder. If `None`, the weights are set to `1.0` for all the adapters.
        """
    if not USE_PEFT_BACKEND:
        raise ValueError('PEFT backend is required for this method.')

    def process_weights(adapter_names, weights):
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)
        if len(adapter_names) != len(weights):
            raise ValueError(
                f'Length of adapter names {len(adapter_names)} is not equal to the length of the weights {len(weights)}'
                )
        weights = [(w if w is not None else 1.0) for w in weights]
        return weights
    adapter_names = [adapter_names] if isinstance(adapter_names, str
        ) else adapter_names
    text_encoder_weights = process_weights(adapter_names, text_encoder_weights)
    text_encoder = text_encoder or getattr(self, 'text_encoder', None)
    if text_encoder is None:
        raise ValueError(
            'The pipeline does not have a default `pipe.text_encoder` class. Please make sure to pass a `text_encoder` instead.'
            )
    set_weights_and_activate_adapters(text_encoder, adapter_names,
        text_encoder_weights)
