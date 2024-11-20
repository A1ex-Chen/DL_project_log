def get_input_embeddings(self) ->nn.Module:
    return self.lang_encoder.get_input_embeddings()
