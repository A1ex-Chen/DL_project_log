def resize_token_embeddings(self, new_num_tokens=None):
    """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
        Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add newly initialized vectors at the end. Reducing the size will remove vectors from the end. 
                If not provided or None: does nothing and just returns a pointer to the input tokens ``tf.Variable`` Module of the model.

        Return: ``tf.Variable``
            Pointer to the input tokens Embeddings Module of the model
        """
    raise NotImplementedError
