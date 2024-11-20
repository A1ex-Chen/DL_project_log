def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
    """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
    if new_num_tokens is None:
        return old_embeddings
    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    if old_num_tokens == new_num_tokens:
        return old_embeddings
    new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings.to(old_embeddings.weight.device)
    self._init_weights(new_embeddings)
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    new_embeddings.weight.data[:num_tokens_to_copy, :
        ] = old_embeddings.weight.data[:num_tokens_to_copy, :]
    return new_embeddings
