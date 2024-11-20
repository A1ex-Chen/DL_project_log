def resize_token_embeddings(self, new_num_tokens: Optional[int]=None
    ) ->torch.nn.Embedding:
    """
        Resizes input token embeddings matrix of the model if :obj:`new_num_tokens != config.vocab_size`.

        Takes care of tying weights embeddings afterwards if the model class has a :obj:`tie_weights()` method.

        Arguments:
            new_num_tokens (:obj:`int`, `optional`):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or :obj:`None`,
                just returns a pointer to the input tokens :obj:`torch.nn.Embedding` module of the model without doing
                anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
    base_model = getattr(self, self.base_model_prefix, self)
    model_embeds = base_model._resize_token_embeddings(new_num_tokens)
    if new_num_tokens is None:
        return model_embeds
    self.config.vocab_size = new_num_tokens
    base_model.vocab_size = new_num_tokens
    self.tie_weights()
    return model_embeds
