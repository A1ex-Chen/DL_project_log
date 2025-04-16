def resize_token_embeddings(self, new_num_tokens=None):
    """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end.
                If not provided or None: does nothing and just returns a pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
    base_model = getattr(self, self.base_model_prefix, self)
    model_embeds = base_model._resize_token_embeddings(new_num_tokens)
    if new_num_tokens is None:
        return model_embeds
    self.config.vocab_size = new_num_tokens
    base_model.vocab_size = new_num_tokens
    if hasattr(self, 'tie_weights'):
        self.tie_weights()
    return model_embeds
