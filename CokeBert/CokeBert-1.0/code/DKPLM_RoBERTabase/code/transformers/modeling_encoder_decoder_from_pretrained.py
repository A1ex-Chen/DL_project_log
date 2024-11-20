@classmethod
def from_pretrained(cls, *args, **kwargs):
    if kwargs.get('decoder_model', None) is None:
        if 'decoder_config' not in kwargs:
            raise ValueError(
                "To load an LSTM in Encoder-Decoder model, please supply either:     - a torch.nn.LSTM model as `decoder_model` parameter (`decoder_model=lstm_model`), or    - a dictionary of configuration parameters that will be used to initialize a      torch.nn.LSTM model as `decoder_config` keyword argument.       E.g. `decoder_config={'input_size': 768, 'hidden_size': 768, 'num_layers': 2}`"
                )
        kwargs['decoder_model'] = torch.nn.LSTM(kwargs.pop('decoder_config'))
    model = super(Model2LSTM, cls).from_pretrained(*args, **kwargs)
    return model
