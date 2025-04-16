def create_and_check_double_heads(self, config, input_ids, token_type_ids,
    position_ids, mc_labels, lm_labels, mc_token_ids):
    model = self.double_head_model_class(config)
    model.eval()
    outputs = model(input_ids, mc_token_ids, lm_labels=lm_labels, mc_labels
        =mc_labels, token_type_ids=token_type_ids, position_ids=position_ids)
    lm_loss, mc_loss, lm_logits, mc_logits = outputs[:4]
    loss = [lm_loss, mc_loss]
    total_voc = self.vocab_size
    self.parent.assertListEqual(list(lm_logits.size()), [self.batch_size,
        self.n_choices, self.seq_length, total_voc])
    self.parent.assertListEqual(list(mc_logits.size()), [self.batch_size,
        self.n_choices])
    self.parent.assertListEqual([list(l.size()) for l in loss], [[], []])
