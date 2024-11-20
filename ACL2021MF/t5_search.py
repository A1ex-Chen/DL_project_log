@torch.no_grad()
def search(self, input_ids: Optional[torch.LongTensor]=None, max_length:
    Optional[int]=None, min_length: Optional[int]=None, do_sample: Optional
    [bool]=None, early_stopping: Optional[bool]=None, num_beams: Optional[
    int]=None, temperature: Optional[float]=None, top_k: Optional[int]=None,
    top_p: Optional[float]=None, repetition_penalty: Optional[float]=None,
    state_transition: Optional[torch.Tensor]=None, bad_words_ids: Optional[
    Iterable[int]]=None, bos_token_id: Optional[int]=None, pad_token_id:
    Optional[int]=None, eos_token_id: Optional[int]=None, length_penalty:
    Optional[float]=None, no_repeat_ngram_size: Optional[int]=None,
    num_return_sequences: Optional[int]=None, decoder_start_token_id:
    Optional[int]=None, use_cache: Optional[bool]=None, **model_kwargs
    ) ->torch.LongTensor:
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
    use_cache = use_cache if use_cache is not None else self.config.use_cache
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
        eos_token_id)
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
        if state_transition is not None:
            beam_scorer = ConstrainedBeamSearchScorer(batch_size=batch_size,
                max_length=max_length, num_beams=num_beams, num_states=
                state_transition.size(1), device=self.device,
                length_penalty=length_penalty, do_early_stopping=
                early_stopping, num_beam_hyps_to_keep=num_return_sequences)
        else:
            beam_scorer = BeamSearchScorer(batch_size=batch_size,
                max_length=max_length, num_beams=num_beams, device=self.
                device, length_penalty=length_penalty, do_early_stopping=
                early_stopping, num_beam_hyps_to_keep=num_return_sequences)
        expand_size = num_beams
        if state_transition is not None:
            expand_size = expand_size * state_transition.size(1)
        input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids,
            expand_size=expand_size, is_encoder_decoder=self.config.
            is_encoder_decoder, **model_kwargs)
        return self.beam_search(input_ids, beam_scorer, logits_processor=
            logits_processor, max_length=max_length, pad_token_id=
            pad_token_id, eos_token_id=eos_token_id, state_transition=
            state_transition, **model_kwargs)
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
