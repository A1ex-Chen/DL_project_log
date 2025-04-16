def create_and_check_bert_for_multiple_choice(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    config.num_choices = self.num_choices
    model = TFBertForMultipleChoice(config=config)
    multiple_choice_inputs_ids = tf.tile(tf.expand_dims(input_ids, 1), (1,
        self.num_choices, 1))
    multiple_choice_input_mask = tf.tile(tf.expand_dims(input_mask, 1), (1,
        self.num_choices, 1))
    multiple_choice_token_type_ids = tf.tile(tf.expand_dims(token_type_ids,
        1), (1, self.num_choices, 1))
    inputs = {'input_ids': multiple_choice_inputs_ids, 'attention_mask':
        multiple_choice_input_mask, 'token_type_ids':
        multiple_choice_token_type_ids}
    logits, = model(inputs)
    result = {'logits': logits.numpy()}
    self.parent.assertListEqual(list(result['logits'].shape), [self.
        batch_size, self.num_choices])
