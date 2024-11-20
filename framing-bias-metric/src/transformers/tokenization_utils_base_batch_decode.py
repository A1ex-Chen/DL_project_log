def batch_decode(self, sequences: Union[List[int], List[List[int]],
    'np.ndarray', 'torch.Tensor', 'tf.Tensor'], skip_special_tokens: bool=
    False, clean_up_tokenization_spaces: bool=True, **kwargs) ->List[str]:
    """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (:obj:`Union[List[int], List[List[int]], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the ``__call__`` method.
            skip_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not to clean up the tokenization spaces.
            kwargs (additional keyword arguments, `optional`):
                Will be passed to the underlying model specific decode method.

        Returns:
            :obj:`List[str]`: The list of decoded sentences.
        """
    return [self.decode(seq, skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs
        ) for seq in sequences]
