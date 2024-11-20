def create_and_check_openai_gpt_double_head(self, config, input_ids,
    input_mask, head_mask, token_type_ids, mc_token_ids, *args):
    model = TFOpenAIGPTDoubleHeadsModel(config=config)
    multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1,
        self.num_choices, 1))
    multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1,
        self.num_choices, 1))
    multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids,
        1), (1, self.num_choices, 1))
    inputs = {'input_ids': multiple_choice_inputs_ids, 'mc_token_ids':
        mc_token_ids, 'attention_mask': multiple_choice_input_mask,
        'token_type_ids': multiple_choice_token_type_ids}
    lm_logits, mc_logits = model(inputs)[:2]
    result = {'lm_logits': lm_logits.numpy(), 'mc_logits': mc_logits.numpy()}
    self.parent.assertListEqual(list(result['lm_logits'].shape), [self.
        batch_size, self.num_choices, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(result['mc_logits'].shape), [self.
        batch_size, self.num_choices])
