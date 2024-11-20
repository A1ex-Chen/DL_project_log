def sample(self, input_ids: torch.LongTensor, logits_processor: Optional[
    LogitsProcessorList]=None, logits_warper: Optional[LogitsProcessorList]
    =None, max_length: Optional[int]=None, pad_token_id: Optional[int]=None,
    eos_token_id: Optional[int]=None, **model_kwargs):
    """
        Generates sequences for models with a language modeling head using multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            logits_processor (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsProcessor` used to modify the prediction scores of the language modeling
                head applied at each generation step.
            logits_warper (:obj:`LogitsProcessorList`, `optional`):
                An instance of :class:`~transformers.LogitsProcessorList`. List of instances of class derived from
                :class:`~transformers.LogitsWarper` used to warp the prediction score distribution of the language
                modeling head applied before multinomial sampling at each generation step.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model. If
                model is an encoder-decoder model the kwargs should include :obj:`encoder_outputs`.

        Return:
            :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`: The generated
            sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter if all
            batches finished early due to the :obj:`eos_token_id`.

        Examples::

            >>> from transformers import (
            ...    AutoTokenizer,
            ...    AutoModelForCausalLM,
            ...    LogitsProcessorList,
            ...    MinLengthLogitsProcessor,
            ...    TopKLogitsWarper,
            ...    TemperatureLogitsWarper,
            ... )

            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

            >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
            >>> model.config.pad_token_id = model.config.eos_token_id

            >>> input_prompt = "Today is a beautiful day, and"
            >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper)

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
    logits_processor = (logits_processor if logits_processor is not None else
        LogitsProcessorList())
    logits_warper = (logits_warper if logits_warper is not None else
        LogitsProcessorList())
    max_length = (max_length if max_length is not None else self.config.
        max_length)
    pad_token_id = (pad_token_id if pad_token_id is not None else self.
        config.pad_token_id)
    eos_token_id = (eos_token_id if eos_token_id is not None else self.
        config.eos_token_id)
    sequence_lengths, unfinished_sequences, cur_len = (self.
        _init_sequence_length_for_generation(input_ids, max_length))
    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(input_ids, **
            model_kwargs)
        outputs = self(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        scores = logits_processor(input_ids, next_token_logits)
        scores = logits_warper(input_ids, scores)
        probs = F.softmax(scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        if eos_token_id is not None:
            assert pad_token_id is not None, 'If eos_token_id is defined, make sure that pad_token_id is defined.'
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        cur_len = cur_len + 1
        if eos_token_id is not None:
            sequence_lengths, unfinished_sequences = (self.
                _update_seq_length_for_generation(sequence_lengths,
                unfinished_sequences, cur_len, next_tokens == eos_token_id))
        if unfinished_sequences.max() == 0:
            break
        model_kwargs = self._update_model_kwargs_for_generation(outputs,
            model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
    return input_ids
