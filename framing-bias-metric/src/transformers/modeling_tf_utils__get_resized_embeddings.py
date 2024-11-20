def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None
    ) ->tf.Variable:
    """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`tf.Variable`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`tf.Variable`` module of the model without doing anything.

        Return:
            :obj:`tf.Variable`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
    word_embeddings = self._get_word_embeddings(old_embeddings)
    if new_num_tokens is None:
        return word_embeddings
    old_num_tokens, old_embedding_dim = word_embeddings.shape
    if old_num_tokens == new_num_tokens:
        return word_embeddings
    init_range = getattr(self.config, 'initializer_range', 0.02)
    new_embeddings = self.add_weight('weight', shape=[new_num_tokens,
        old_embedding_dim], initializer=get_initializer(init_range), dtype=
        tf.float32)
    init_weights = new_embeddings.numpy()
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    init_weights[:num_tokens_to_copy] = word_embeddings[:num_tokens_to_copy, :]
    new_embeddings.assign(init_weights)
    return new_embeddings
