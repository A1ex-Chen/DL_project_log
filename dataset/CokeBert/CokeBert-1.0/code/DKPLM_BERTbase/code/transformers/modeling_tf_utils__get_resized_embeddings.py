def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
    """ Build a resized Embedding Variable from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``tf.Variable``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
