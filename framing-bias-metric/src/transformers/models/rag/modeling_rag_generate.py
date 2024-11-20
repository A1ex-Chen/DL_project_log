@torch.no_grad()
def generate(self, input_ids: Optional[torch.LongTensor]=None,
    attention_mask: Optional[torch.LongTensor]=None, context_input_ids=None,
    context_attention_mask=None, doc_scores=None, max_length=None,
    min_length=None, early_stopping=None, use_cache=None, num_beams=None,
    bos_token_id=None, pad_token_id=None, eos_token_id=None, length_penalty
    =None, no_repeat_ngram_size=None, repetition_penalty=None,
    bad_words_ids=None, num_return_sequences=None, decoder_start_token_id=
    None, n_docs=None, prefix_allowed_tokens_fn: Callable[[int, torch.
    Tensor], List[int]]=None, **model_kwargs):
    """
        Implements RAG token decoding.

        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`input_ids` is not passed, then
                :obj:`context_input_ids` has to be provided.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            context_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Input IDs post-processed from the retrieved documents and the question encoder :obj:`input_ids` by the
                retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            context_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size * config.n_docs, config.max_combined_length)`, `optional`, returned when `output_retrieved=True`):
                Attention mask post-processed from the retrieved documents and the question encoder :obj:`input_ids` by
                the retriever.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            doc_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.n_docs)`):
                Score between each retrieved document embeddings (see :obj:`retrieved_doc_embeds`) and
                :obj:`question_encoder_last_hidden_state`.

                If the model has is not initialized with a ``retriever``, :obj:`context_input_ids` has to be provided
                to the forward pass. :obj:`context_input_ids` are returned by
                :meth:`~transformers.RagRetriever.__call__`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to stop the beam search when at least ``num_beams`` sentences are finished per batch or
                not.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch. Note that this
                is not the value we pass to the ``generator``'s `:func:`~transformers.PreTrainedModel.generate`
                function, where we set ``num_return_sequences`` to :obj:`num_beams`.
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            n_docs (:obj:`int`, `optional`, defaults to :obj:`config.n_docs`)
                Number of documents to retrieve and/or number of documents for which to generate an answer.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
                :obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.
        """
    n_docs = n_docs if n_docs is not None else self.config.n_docs
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    max_length = (max_length if max_length is not None else self.config.
        max_length)
    num_return_sequences = (num_return_sequences if num_return_sequences is not
        None else self.config.num_return_sequences)
    bos_token_id = (bos_token_id if bos_token_id is not None else self.
        config.generator.bos_token_id)
    eos_token_id = (eos_token_id if eos_token_id is not None else self.
        config.generator.eos_token_id)
    pad_token_id = (pad_token_id if pad_token_id is not None else self.
        config.generator.pad_token_id)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    decoder_start_token_id = (decoder_start_token_id if 
        decoder_start_token_id is not None else self.config.generator.
        decoder_start_token_id)
    if self.retriever is not None and context_input_ids is None:
        question_hidden_states = self.question_encoder(input_ids,
            attention_mask=attention_mask)[0]
        out = self.retriever(input_ids, question_hidden_states.cpu().detach
            ().to(torch.float32).numpy(), prefix=self.generator.config.
            prefix, n_docs=n_docs, return_tensors='pt')
        context_input_ids, context_attention_mask, retrieved_doc_embeds = out[
            'context_input_ids'], out['context_attention_mask'], out[
            'retrieved_doc_embeds']
        retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
        context_input_ids = context_input_ids.to(input_ids)
        context_attention_mask = context_attention_mask.to(input_ids)
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1),
            retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
    assert context_input_ids.shape[0
        ] % n_docs == 0, f' The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}.'
    batch_size = context_input_ids.shape[0] // n_docs
    encoder = self.rag.generator.get_encoder()
    encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=
        context_attention_mask)
    input_ids = torch.full((batch_size * num_beams, 1),
        decoder_start_token_id, dtype=torch.long, device=next(self.
        parameters()).device)
    last_hidden_state = encoder_outputs['last_hidden_state']

    def extend_enc_output(tensor, num_beams=None):
        tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) +
            tensor.shape[1:])
        tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.
            shape[3:])
        return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.
            shape[3:])
    context_attention_mask = extend_enc_output(context_attention_mask,
        num_beams=num_beams)
    encoder_outputs['last_hidden_state'] = extend_enc_output(last_hidden_state,
        num_beams=num_beams)
    doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)
    model_kwargs['doc_scores'] = doc_scores
    model_kwargs['encoder_outputs'] = encoder_outputs
    model_kwargs['attention_mask'] = context_attention_mask
    model_kwargs['n_docs'] = n_docs
    pre_processor = self._get_logits_processor(repetition_penalty=
        repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids, min_length=min_length, eos_token_id=
        eos_token_id, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams)
    if num_beams == 1:
        if num_return_sequences > 1:
            raise ValueError(
                f'num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search.'
                )
        return self.greedy_search(input_ids, pre_processor=pre_processor,
            max_length=max_length, pad_token_id=pad_token_id, eos_token_id=
            eos_token_id, **model_kwargs)
    elif num_beams > 1:
        length_penalty = (length_penalty if length_penalty is not None else
            self.config.length_penalty)
        early_stopping = (early_stopping if early_stopping is not None else
            self.config.early_stopping)
        if num_return_sequences > num_beams:
            raise ValueError(
                '`num_return_sequences` has to be smaller or equal to `num_beams`.'
                )
        beam_scorer = BeamSearchScorer(batch_size=batch_size, max_length=
            max_length, num_beams=num_beams, device=self.device,
            length_penalty=length_penalty, do_early_stopping=early_stopping,
            num_beam_hyps_to_keep=num_return_sequences)
        return self.beam_search(input_ids, beam_scorer, pre_processor=
            pre_processor, max_length=max_length, pad_token_id=pad_token_id,
            eos_token_id=eos_token_id, **model_kwargs)
    else:
        raise ValueError(
            f'`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}'
            )
