def _generate_beam_search(self, input_ids, cur_len, max_length, min_length,
    do_sample, early_stopping, temperature, top_k, top_p,
    repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id,
    eos_token_id, batch_size, num_return_sequences, length_penalty,
    num_beams, vocab_size, encoder_outputs, attention_mask, use_cache):
    """Generate sequences for each example with beam search."""
    generated_hyps = [BeamHypotheses(num_beams, max_length, length_penalty,
        early_stopping=early_stopping) for _ in range(batch_size)]
    if do_sample is False:
        beam_scores_begin = tf.zeros((batch_size, 1), dtype=tf.float32)
        beam_scores_end = tf.ones((batch_size, num_beams - 1), dtype=tf.float32
            ) * -1000000000.0
        beam_scores = tf.concat([beam_scores_begin, beam_scores_end], -1)
    else:
        beam_scores = tf.zeros((batch_size, num_beams), dtype=tf.float32)
    beam_scores = tf.reshape(beam_scores, (batch_size * num_beams,))
    past = encoder_outputs
    done = [(False) for _ in range(batch_size)]
    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(input_ids, past=
            past, attention_mask=attention_mask, use_cache=use_cache)
        outputs = self(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]
        if self._use_cache(outputs, use_cache):
            past = outputs[1]
        if repetition_penalty != 1.0:
            next_token_logits_penalties = _create_next_token_logits_penalties(
                input_ids, next_token_logits, repetition_penalty)
            next_token_logits = tf.math.multiply(next_token_logits,
                next_token_logits_penalties)
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        if self.config.is_encoder_decoder and do_sample is False:
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length)
        scores = tf.nn.log_softmax(next_token_logits, axis=-1)
        if eos_token_id is not None and cur_len < min_length:
            num_batch_hypotheses = batch_size * num_beams
            is_token_logit_eos_token = tf.convert_to_tensor([(True if token is
                eos_token_id else False) for token in range(vocab_size)],
                dtype=tf.bool)
            eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token,
                [num_batch_hypotheses, vocab_size])
            scores = set_tensor_by_indices_to_value(scores,
                eos_token_indices_mask, -float('inf'))
        if no_repeat_ngram_size > 0:
            num_batch_hypotheses = batch_size * num_beams
            banned_tokens = calc_banned_ngram_tokens(input_ids,
                num_batch_hypotheses, no_repeat_ngram_size, cur_len)
            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append([(True if token in
                    banned_tokens_slice else False) for token in range(
                    vocab_size)])
            scores = set_tensor_by_indices_to_value(scores, tf.
                convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool
                ), -float('inf'))
        if bad_words_ids is not None:
            banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)
            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append([(True if token in
                    banned_tokens_slice else False) for token in range(
                    vocab_size)])
            scores = set_tensor_by_indices_to_value(scores, tf.
                convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool
                ), -float('inf'))
        assert shape_list(scores) == [batch_size * num_beams, vocab_size]
        if do_sample:
            _scores = scores + tf.broadcast_to(beam_scores[:, None], (
                batch_size * num_beams, vocab_size))
            _scores = tf_top_k_top_p_filtering(_scores, top_k=top_k, top_p=
                top_p, min_tokens_to_keep=2)
            _scores = tf.reshape(_scores, (batch_size, num_beams * vocab_size))
            next_tokens = sample_without_replacement(_scores, num_samples=2 *
                num_beams)
            next_scores = tf.gather(_scores, next_tokens, batch_dims=1)
            next_scores_indices = tf.argsort(next_scores, direction=
                'DESCENDING', axis=1)
            next_scores = tf.gather(next_scores, next_scores_indices,
                batch_dims=1)
            next_tokens = tf.gather(next_tokens, next_scores_indices,
                batch_dims=1)
        else:
            next_scores = scores + tf.broadcast_to(beam_scores[:, None], (
                batch_size * num_beams, vocab_size))
            next_scores = tf.reshape(next_scores, (batch_size, num_beams *
                vocab_size))
            next_scores, next_tokens = tf.math.top_k(next_scores, k=2 *
                num_beams, sorted=True)
        assert shape_list(next_scores) == shape_list(next_tokens) == [
            batch_size, 2 * num_beams]
        next_batch_beam = []
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                assert len(generated_hyps[batch_idx]
                    ) >= num_beams, 'Batch can only be done if at least {} beams have been generated'.format(
                    num_beams)
                assert eos_token_id is not None and pad_token_id is not None, 'generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)
                continue
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score
                ) in enumerate(zip(next_tokens[batch_idx], next_scores[
                batch_idx])):
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size
                effective_beam_id = batch_idx * num_beams + beam_id
                if eos_token_id is not None and token_id.numpy(
                    ) == eos_token_id:
                    is_beam_token_worse_than_top_num_beams = (
                        beam_token_rank >= num_beams)
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(tf.identity(input_ids[
                        effective_beam_id]), beam_token_score.numpy())
                else:
                    next_sent_beam.append((beam_token_score, token_id,
                        effective_beam_id))
                if len(next_sent_beam) == num_beams:
                    break
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx
                ].is_done(tf.reduce_max(next_scores[batch_idx]).numpy(),
                cur_len)
            assert len(next_sent_beam
                ) == num_beams, 'Beam should always be full'
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)
        if all(done):
            break
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = tf.convert_to_tensor([x[0] for x in next_batch_beam],
            dtype=tf.float32)
        beam_tokens = tf.convert_to_tensor([x[1] for x in next_batch_beam],
            dtype=tf.int32)
        beam_idx = tf.convert_to_tensor([x[2] for x in next_batch_beam],
            dtype=tf.int32)
        input_ids = tf.stack([tf.identity(input_ids[x, :]) for x in beam_idx])
        input_ids = tf.concat([input_ids, tf.expand_dims(beam_tokens, 1)],
            axis=-1)
        cur_len = cur_len + 1
        if past is not None:
            past = self._reorder_cache(past, beam_idx)
        if self.config.is_encoder_decoder is False:
            attention_mask = tf.concat([attention_mask, tf.ones((shape_list
                (attention_mask)[0], 1), dtype=tf.int32)], axis=-1)
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue
        if eos_token_id is not None and all((token_id % vocab_size).numpy()
            .item() != eos_token_id for token_id in next_tokens[batch_idx]):
            assert tf.reduce_all(next_scores[batch_idx, :num_beams] == tf.
                reshape(beam_scores, (batch_size, num_beams))[batch_idx]
                ), 'If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}'.format(
                next_scores[:, :num_beams][batch_idx], tf.reshape(
                beam_scores, (batch_size, num_beams))[batch_idx])
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].numpy().item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)
    output_batch_size = (batch_size if do_sample else batch_size *
        num_return_sequences)
    output_num_return_sequences_per_batch = (1 if do_sample else
        num_return_sequences)
    sent_lengths_list = []
    best = []
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths_list.append(len(best_hyp))
            best.append(best_hyp)
    assert output_batch_size == len(best
        ), 'Output batch size {} must match output beam hypotheses {}'.format(
        output_batch_size, len(best))
    sent_lengths = tf.convert_to_tensor(sent_lengths_list, dtype=tf.int32)
    if tf.reduce_min(sent_lengths).numpy() != tf.reduce_max(sent_lengths
        ).numpy():
        assert pad_token_id is not None, '`Pad_token_id` has to be defined'
        sent_max_len = min(tf.reduce_max(sent_lengths).numpy() + 1, max_length)
        decoded_list = []
        for i, hypo in enumerate(best):
            assert sent_lengths[i] == shape_list(hypo)[0]
            if sent_lengths[i] == sent_max_len:
                decoded_slice = hypo
            else:
                num_pad_tokens = sent_max_len - sent_lengths[i]
                padding = pad_token_id * tf.ones((num_pad_tokens,), dtype=
                    tf.int32)
                decoded_slice = tf.concat([hypo, padding], axis=-1)
                if sent_lengths[i] < max_length:
                    decoded_slice = tf.where(tf.range(sent_max_len, dtype=
                        tf.int32) == sent_lengths[i], eos_token_id * tf.
                        ones((sent_max_len,), dtype=tf.int32), decoded_slice)
            decoded_list.append(decoded_slice)
        decoded = tf.stack(decoded_list)
    else:
        assert (len(hypo) == max_length for hypo in best)
        decoded = tf.stack(best)
    return decoded
