def __call__(self, question_input_ids: List[List[int]],
    question_hidden_states: np.ndarray, prefix=None, n_docs=None,
    return_tensors=None) ->BatchEncoding:
    """
        Retrieves documents for specified :obj:`question_hidden_states`.

        Args:
            question_input_ids: (:obj:`List[List[int]]`) batch of input ids
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (:obj:`str`, `optional`):
                The prefix used by the generator's tokenizer.
            n_docs (:obj:`int`, `optional`):
                The number of docs retrieved per query.
            return_tensors (:obj:`str` or :class:`~transformers.tokenization_utils_base.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.

        Returns: :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following
        fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """
    n_docs = n_docs if n_docs is not None else self.n_docs
    prefix = prefix if prefix is not None else self.config.generator.prefix
    retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states,
        n_docs)
    input_strings = self.question_encoder_tokenizer.batch_decode(
        question_input_ids, skip_special_tokens=True)
    context_input_ids, context_attention_mask = self.postprocess_docs(docs,
        input_strings, prefix, n_docs, return_tensors=return_tensors)
    return BatchEncoding({'context_input_ids': context_input_ids,
        'context_attention_mask': context_attention_mask,
        'retrieved_doc_embeds': retrieved_doc_embeds, 'doc_ids': doc_ids},
        tensor_type=return_tensors)
