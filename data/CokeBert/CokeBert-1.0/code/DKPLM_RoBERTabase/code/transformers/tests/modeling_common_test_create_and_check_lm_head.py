def create_and_check_lm_head(self, config, input_ids, token_type_ids,
    position_ids, mc_labels, lm_labels, mc_token_ids):
    model = self.lm_head_model_class(config)
    model.eval()
    outputs = model(input_ids, position_ids, token_type_ids, lm_labels)
    loss, lm_logits = outputs[:2]
    total_voc = self.vocab_size
    self.parent.assertListEqual(list(lm_logits.size()), [self.batch_size,
        self.n_choices, self.seq_length, total_voc])
    self.parent.assertListEqual(list(loss.size()), [])
