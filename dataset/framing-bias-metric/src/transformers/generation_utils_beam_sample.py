def beam_sample(self, input_ids: torch.LongTensor, beam_scorer: BeamScorer,
    logits_processor: Optional[LogitsProcessorList]=None, logits_warper:
    Optional[LogitsProcessorList]=None, max_length: Optional[int]=None,
    pad_token_id: Optional[int]=None, eos_token_id: Optional[int]=None, **
    model_kwargs):
    """
        Generates sequences for models with a language modeling head using beam search with multinomial sampling.

        Parameters:

            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes it as an empty
                :obj:`torch.LongTensor` of shape :obj:`(1,)`.
            beam_scorer (:obj:`BeamScorer`):
                A derived instance of :class:`~transformers.BeamScorer` that defines how beam hypotheses are
                constructed, stored and sorted during generation. For more information, the documentation of
                :class:`~transformers.BeamScorer` should be read.
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
            ...     AutoTokenizer,
            ...     AutoModelForSeq2SeqLM,
            ...     LogitsProcessorList,
            ...     MinLengthLogitsProcessor,
            ...     TopKLogitsWarper,
            ...     TemperatureLogitsWarper,
            ...     BeamSearchScorer,
            ... )
            >>> import torch

            >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
            >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

            >>> encoder_input_str = "translate English to German: How old are you?"
            >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

            >>> # lets run beam search using 3 beams
            >>> num_beams = 3
            >>> # define decoder start token ids
            >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
            >>> input_ids = input_ids * model.config.decoder_start_token_id

            >>> # add encoder_outputs to model keyword arguments
            >>> model_kwargs = {
            ...     "encoder_outputs": model.get_encoder()(encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True)
            ... }

            >>> # instantiate beam scorer
            >>> beam_scorer = BeamSearchScorer(
            ...     batch_size=1,
            ...     max_length=model.config.max_length,
            ...     num_beams=num_beams,
            ...     device=model.device,
            ... )

            >>> # instantiate logits processors
            >>> logits_processor = LogitsProcessorList([
            ...     MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)
            ... ])
            >>> # instantiate logits processors
            >>> logits_warper = LogitsProcessorList([
            ...     TopKLogitsWarper(50),
            ...     TemperatureLogitsWarper(0.7),
            ... ])

            >>> outputs = model.beam_sample(
            ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
            ... )

            >>> print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))
        """
    logits_processor = (logits_processor if logits_processor is not None else
        LogitsProcessorList())
    max_length = (max_length if max_length is not None else self.config.
        max_length)
    pad_token_id = (pad_token_id if pad_token_id is not None else self.
        config.pad_token_id)
    eos_token_id = (eos_token_id if eos_token_id is not None else self.
        config.eos_token_id)
    batch_size = len(beam_scorer._beam_hyps)
    num_beams = beam_scorer.num_beams
    batch_beam_size, cur_len = input_ids.shape
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float,
        device=input_ids.device)
    beam_scores = beam_scores.view((batch_size * num_beams,))
    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(input_ids, **
            model_kwargs)
        outputs = self(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len, max_length=max_length)
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        next_token_scores = logits_processor(input_ids, next_token_scores)
        next_token_scores = next_token_scores + beam_scores[:, None].expand_as(
            next_token_scores)
        next_token_scores = logits_warper(input_ids, next_token_scores)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size, num_beams *
            vocab_size)
        probs = F.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
        next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
        next_token_scores, _indices = torch.sort(next_token_scores,
            descending=True, dim=1)
        next_tokens = torch.gather(next_tokens, -1, _indices)
        next_indices = next_tokens // vocab_size
        next_tokens = next_tokens % vocab_size
        beam_outputs = beam_scorer.process(input_ids, next_token_scores,
            next_tokens, next_indices, pad_token_id=pad_token_id,
            eos_token_id=eos_token_id)
        beam_scores = beam_outputs['next_beam_scores']
        beam_next_tokens = beam_outputs['next_beam_tokens']
        beam_idx = beam_outputs['next_beam_indices']
        input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.
            unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1
        model_kwargs = self._update_model_kwargs_for_generation(outputs,
            model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder)
        if model_kwargs['past'] is not None:
            model_kwargs['past'] = self._reorder_cache(model_kwargs['past'],
                beam_idx)
        if beam_scorer.is_done:
            break
    decoded = beam_scorer.finalize(input_ids, beam_scores, next_tokens,
        next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
    return decoded
