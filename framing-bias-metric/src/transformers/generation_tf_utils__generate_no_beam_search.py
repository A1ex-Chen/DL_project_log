def _generate_no_beam_search(self, input_ids, cur_len, max_length,
    min_length, do_sample, temperature, top_k, top_p, repetition_penalty,
    no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id,
    batch_size, vocab_size, encoder_outputs, attention_mask, use_cache):
    """
        Generate sequences for each example without beam search (num_beams == 1). All returned sequence are generated
        independantly.
        """
    unfinished_sents = tf.ones_like(input_ids[:, 0])
    sent_lengths = tf.ones_like(input_ids[:, 0]) * max_length
    past = encoder_outputs
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
        if no_repeat_ngram_size > 0:
            banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size,
                no_repeat_ngram_size, cur_len)
            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append([(True if token in
                    banned_tokens_slice else False) for token in range(
                    vocab_size)])
            next_token_logits = set_tensor_by_indices_to_value(
                next_token_logits, tf.convert_to_tensor(
                banned_tokens_indices_mask, dtype=tf.bool), -float('inf'))
        if bad_words_ids is not None:
            banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)
            banned_tokens_indices_mask = []
            for banned_tokens_slice in banned_tokens:
                banned_tokens_indices_mask.append([(True if token in
                    banned_tokens_slice else False) for token in range(
                    vocab_size)])
            next_token_logits = set_tensor_by_indices_to_value(
                next_token_logits, tf.convert_to_tensor(
                banned_tokens_indices_mask, dtype=tf.bool), -float('inf'))
        if eos_token_id is not None and cur_len < min_length:
            is_token_logit_eos_token = tf.convert_to_tensor([(True if token is
                eos_token_id else False) for token in range(vocab_size)],
                dtype=tf.bool)
            eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token,
                [batch_size, vocab_size])
            next_token_logits = set_tensor_by_indices_to_value(
                next_token_logits, eos_token_indices_mask, -float('inf'))
        if do_sample:
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            next_token_logits = tf_top_k_top_p_filtering(next_token_logits,
                top_k=top_k, top_p=top_p)
            next_token = tf.squeeze(tf.random.categorical(next_token_logits,
                dtype=tf.int32, num_samples=1), axis=1)
        else:
            next_token = tf.math.argmax(next_token_logits, axis=-1,
                output_type=tf.int32)
        if eos_token_id is not None:
            tokens_to_add = next_token * unfinished_sents + pad_token_id * (
                1 - unfinished_sents)
        else:
            tokens_to_add = next_token
        input_ids = tf.concat([input_ids, tf.expand_dims(tokens_to_add, -1)], 1
            )
        cur_len = cur_len + 1
        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            is_sents_unfinished_and_token_to_add_is_eos = tf.math.multiply(
                unfinished_sents, tf.cast(eos_in_sents, tf.int32))
            sent_lengths = sent_lengths * (1 -
                is_sents_unfinished_and_token_to_add_is_eos
                ) + cur_len * is_sents_unfinished_and_token_to_add_is_eos
            unfinished_sents -= is_sents_unfinished_and_token_to_add_is_eos
        if tf.math.reduce_max(unfinished_sents) == 0:
            break
        if self.config.is_encoder_decoder is False:
            attention_mask = tf.concat([attention_mask, tf.ones((shape_list
                (attention_mask)[0], 1), dtype=tf.int32)], axis=-1)
    min_sent_length = tf.math.reduce_min(sent_lengths)
    max_sent_length = tf.math.reduce_max(sent_lengths)
    if min_sent_length != max_sent_length:
        assert pad_token_id is not None, '`Pad_token_id` has to be defined if batches have different lengths'
        padding = tf.ones([batch_size, max_sent_length.numpy()], dtype=tf.int32
            ) * pad_token_id
        broad_casted_sent_lengths = tf.broadcast_to(tf.expand_dims(
            sent_lengths, -1), [batch_size, max_sent_length])
        broad_casted_range = tf.transpose(tf.broadcast_to(tf.expand_dims(tf
            .range(max_sent_length), -1), [max_sent_length, batch_size]))
        decoded = tf.where(broad_casted_range < broad_casted_sent_lengths,
            input_ids, padding)
    else:
        decoded = input_ids
    return decoded
