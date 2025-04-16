@property
def num_hidden_layers(self) ->int:
    return self.num_encoder_layers + self.num_decoder_layers
