def floating_point_ops(self, input_dict: Dict[str, Union[torch.Tensor, Any]
    ], exclude_embeddings: bool=True) ->int:
    """
        Get number of (optionally, non-embeddings) floating-point operations for the forward and backward passes of a
        batch with this transformer model. Default approximation neglects the quadratic dependency on the number of
        tokens (valid if :obj:`12 * d_model << sequence_length`) as laid out in `this paper
        <https://arxiv.org/pdf/2001.08361.pdf>`__ section 2.1. Should be overridden for transformers with parameter
        re-use e.g. Albert or Universal Transformers, or if doing long-range modeling with very high sequence lengths.

        Args:
            batch_size (:obj:`int`):
                The batch size for the forward pass.

            sequence_length (:obj:`int`):
                The number of tokens in each line of the batch.

            exclude_embeddings (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to count embedding and softmax operations.

        Returns:
            :obj:`int`: The number of floating-point operations.
        """
    return 6 * self.estimate_tokens(input_dict) * self.num_parameters(
        exclude_embeddings=exclude_embeddings)
