def create_and_check_double_lm_head_model(self, config, input_ids,
    input_mask, head_mask, token_type_ids, mc_token_ids, *args):
    model = GPT2DoubleHeadsModel(config)
    model.eval()
    multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.
        num_choices, -1).contiguous()
    multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.
        num_choices, -1).contiguous()
    multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1,
        self.num_choices, -1).contiguous()
    inputs = {'input_ids': multiple_choice_inputs_ids, 'mc_token_ids':
        mc_token_ids, 'attention_mask': multiple_choice_input_mask,
        'token_type_ids': multiple_choice_token_type_ids, 'lm_labels':
        multiple_choice_inputs_ids}
    loss, lm_logits, mc_logits, _ = model(**inputs)
    result = {'loss': loss, 'lm_logits': lm_logits, 'mc_logits': mc_logits}
    self.parent.assertListEqual(list(result['loss'].size()), [])
    self.parent.assertListEqual(list(result['lm_logits'].size()), [self.
        batch_size, self.num_choices, self.seq_length, self.vocab_size])
    self.parent.assertListEqual(list(result['mc_logits'].size()), [self.
        batch_size, self.num_choices])
