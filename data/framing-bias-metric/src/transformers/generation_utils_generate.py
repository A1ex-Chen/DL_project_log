@torch.no_grad()
def generate(self, input_ids: Optional[torch.LongTensor]=None, max_length:
    Optional[int]=None, min_length: Optional[int]=None, do_sample: Optional
    [bool]=None, early_stopping: Optional[bool]=None, num_beams: Optional[
    int]=None, temperature: Optional[float]=None, top_k: Optional[int]=None,
    top_p: Optional[float]=None, repetition_penalty: Optional[float]=None,
    bad_words_ids: Optional[Iterable[int]]=None, bos_token_id: Optional[int
    ]=None, pad_token_id: Optional[int]=None, eos_token_id: Optional[int]=
    None, length_penalty: Optional[float]=None, no_repeat_ngram_size:
    Optional[int]=None, num_return_sequences: Optional[int]=None,
    decoder_start_token_id: Optional[int]=None, use_cache: Optional[bool]=
    None, prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
    List[int]]]=None, **model_kwargs) ->torch.LongTensor:
    """
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults tp 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to :obj:`top_p` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the
                model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer
                sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[List[int]]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer(bad_word,
                add_prefix_space=True).input_ids`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens. If not provided, will default to a tensor the same
                shape as :obj:`input_ids` that masks the pad token. `What are attention masks?
                <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            prefix_allowed_tokens_fn: (:obj:`Callable[[int, torch.Tensor], List[int]]`, `optional`):
                If provided, this function constraints the beam search to allowed tokens only at each step. If not
                provided no constraint is applied. This function takes 2 arguments :obj:`inputs_ids` and the batch ID
                :obj:`batch_id`. It has to return a list with the allowed tokens for the next generation step
                conditioned on the previously generated tokens :obj:`inputs_ids` and the batch ID :obj:`batch_id`. This
                argument is useful for constrained generation conditioned on the prefix, as described in
                `Autoregressive Entity Retrieval <https://arxiv.org/abs/2010.00904>`__.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If the
                model is an Encoder-Decoder model, encoder specific kwargs should not be prefixed and decoder specific
                kwargs should be prefixed with `decoder_`.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.

        Examples::

            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> # do greedy decoding without providing a prompt
            >>> outputs = model.generate(max_length=40)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            >>> document = (
            ... "at least two people were killed in a suspected bomb attack on a passenger bus "
            ... "in the strife-torn southern philippines on monday , the military said."
            ... )
            >>> # encode input contex
            >>> input_ids = tokenizer(document, return_tensors="pt").input_ids
            >>> # generate 3 independent sequences using beam search decoding (5 beams)
            >>> # with T5 encoder-decoder model conditioned on short news article.
            >>> outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("distilgpt2")
            >>> input_context = "The dog"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate 3 candidates using sampling
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, num_return_sequences=3, do_sample=True)
            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("ctrl")
            >>> model = AutoModelForCausalLM.from_pretrained("ctrl")
            >>> # "Legal" is one of the control codes for ctrl
            >>> input_context = "Legal My neighbor is"
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, repetition_penalty=1.2)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> input_context = "My cute dog"
            >>> # get tokens of words that should not be generated
            >>> bad_words_ids = [tokenizer(bad_word, add_prefix_space=True).input_ids for bad_word in ["idiot", "stupid", "shut up"]]
            >>> # encode input context
            >>> input_ids = tokenizer(input_context, return_tensors="pt").input_ids
            >>> # generate sequences without allowing bad_words to be generated
            >>> outputs = model.generate(input_ids=input_ids, max_length=20, do_sample=True, bad_words_ids=bad_words_ids)
            >>> print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))
        """
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    max_length = (max_length if max_length is not None else self.config.
        max_length)
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    num_return_sequences = (num_return_sequences if num_return_sequences is not
        None else self.config.num_return_sequences)
    pad_token_id = (pad_token_id if pad_token_id is not None else self.
        config.pad_token_id)
    bos_token_id = (bos_token_id if bos_token_id is not None else self.
        config.bos_token_id)
    eos_token_id = (eos_token_id if eos_token_id is not None else self.
        config.eos_token_id)
    if input_ids is None:
        input_ids = self._prepare_input_ids_for_generation(bos_token_id)
    if model_kwargs.get('attention_mask', None) is None:
        model_kwargs['attention_mask'
            ] = self._prepare_attention_mask_for_generation(input_ids,
            pad_token_id, eos_token_id)
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.'
            )
        pad_token_id = eos_token_id
    if self.config.is_encoder_decoder:
        model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
            input_ids, model_kwargs)
        input_ids = self._prepare_decoder_input_ids_for_generation(input_ids,
            decoder_start_token_id=decoder_start_token_id, bos_token_id=
            bos_token_id, **model_kwargs)
        if 'encoder_outputs' not in model_kwargs or not isinstance(model_kwargs
            ['encoder_outputs'], ModelOutput):
            raise ValueError(
                'Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.'
                )
    is_greedy_gen_mode = num_beams == 1 and do_sample is False
    is_sample_gen_mode = num_beams == 1 and do_sample is True
    is_beam_gen_mode = num_beams > 1 and do_sample is False
    is_beam_sample_gen_mode = num_beams > 1 and do_sample is True
    model_kwargs['use_cache'] = use_cache
    logits_processor = self._get_logits_processor(repetition_penalty=
        repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
        bad_words_ids=bad_words_ids, min_length=min_length, eos_token_id=
        eos_token_id, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        num_beams=num_beams)
    if is_greedy_gen_mode:
        if num_return_sequences > 1:
            raise ValueError(
                f'num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search.'
                )
        return self.greedy_search(input_ids, logits_processor=
            logits_processor, max_length=max_length, pad_token_id=
            pad_token_id, eos_token_id=eos_token_id, **model_kwargs)
    elif is_sample_gen_mode:
        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p,
            temperature=temperature, num_beams=num_beams)
        input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids,
            expand_size=num_return_sequences, is_encoder_decoder=self.
            config.is_encoder_decoder, **model_kwargs)
        return self.sample(input_ids, logits_processor=logits_processor,
            logits_warper=logits_warper, max_length=max_length,
            pad_token_id=pad_token_id, eos_token_id=eos_token_id, **
            model_kwargs)
    elif is_beam_gen_mode:
        batch_size = input_ids.shape[0]
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
        input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids,
            expand_size=num_beams, is_encoder_decoder=self.config.
            is_encoder_decoder, **model_kwargs)
        return self.beam_search(input_ids, beam_scorer, logits_processor=
            logits_processor, max_length=max_length, pad_token_id=
            pad_token_id, eos_token_id=eos_token_id, **model_kwargs)
    elif is_beam_sample_gen_mode:
        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p,
            temperature=temperature, num_beams=num_beams)
        batch_size = input_ids.shape[0] * num_return_sequences
        length_penalty = (length_penalty if length_penalty is not None else
            self.config.length_penalty)
        beam_scorer = BeamSearchScorer(batch_size=batch_size, max_length=
            max_length, num_beams=num_beams, device=self.device,
            length_penalty=length_penalty, do_early_stopping=early_stopping)
        input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids,
            expand_size=num_beams * num_return_sequences,
            is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
        return self.beam_sample(input_ids, beam_scorer, logits_processor=
            logits_processor, logits_warper=logits_warper, max_length=
            max_length, pad_token_id=pad_token_id, eos_token_id=
            eos_token_id, **model_kwargs)
