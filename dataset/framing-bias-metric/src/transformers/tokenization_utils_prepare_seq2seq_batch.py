def prepare_seq2seq_batch(self, src_texts: List[str], tgt_texts: Optional[
    List[str]]=None, max_length: Optional[int]=None, max_target_length:
    Optional[int]=None, padding: str='longest', return_tensors: str='None',
    truncation=True, **kwargs) ->BatchEncoding:
    """

        Prepare a batch that can be passed directly to an instance of :class:`~transformers.AutoModelForSeq2SeqLM`.

        Args:
            src_texts: (:obj:`List[str]`):
                List of documents to summarize or source language texts.
            tgt_texts: (:obj:`List[str]`, `optional`):
                List of summaries or target language texts.
            max_length (:obj:`int`, `optional`):
                Controls the maximum length for encoder inputs (documents to summarize or source language texts). If
                left unset or set to :obj:`None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            max_target_length (:obj:`int`, `optional`):
                Controls the maximum length of decoder inputs (target language texts or summaries). If left unset or
                set to :obj:`None`, this will use the max_length value.
            padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`False`):
                Activates and controls padding. Accepts the following values:

                * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
                  single sequence if provided).
                * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided.
                * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
                  different lengths).
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.
            truncation (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.TruncationStrategy`, `optional`, defaults to :obj:`True`):
                Activates and controls truncation. Accepts the following values:

                * :obj:`True` or :obj:`'longest_first'`: Truncate to a maximum length specified with the argument
                  :obj:`max_length` or to the maximum acceptable input length for the model if that argument is not
                  provided. This will truncate token by token, removing a token from the longest sequence in the pair
                  if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_first'`: Truncate to a maximum length specified with the argument :obj:`max_length` or to
                  the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`'only_second'`: Truncate to a maximum length specified with the argument :obj:`max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                * :obj:`False` or :obj:`'do_not_truncate'` (default): No truncation (i.e., can output batch with
                  sequence lengths greater than the model maximum admissible input size).
            **kwargs:
                Additional keyword arguments passed along to :obj:`self.__call__`.

        Returns:
            :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following fields:

            - **input_ids** -- List of token ids to be fed to the encoder.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
            - **labels** -- List of token ids for tgt_texts

            The full set of keys ``[input_ids, attention_mask, labels]``, will only be returned if tgt_texts is passed.
            Otherwise, input_ids, attention_mask will be the only keys.
        """
    raise NotImplementedError(
        'If your model requires more than input_ids for a typical forward pass, you should implement this method. Returned keys should be [input_ids, attention_mask, labels]. See MarianTokenizer or T5Tokenizer for a reference implementation.'
        )
