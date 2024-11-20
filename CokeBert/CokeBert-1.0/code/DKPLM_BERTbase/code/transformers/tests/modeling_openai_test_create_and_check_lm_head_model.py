def create_and_check_lm_head_model(self, config, input_ids, head_mask,
    token_type_ids, *args):
    model = OpenAIGPTLMHeadModel(config)
    model.eval()
    loss, lm_logits = model(input_ids, token_type_ids=token_type_ids,
        labels=input_ids)
    result = {'loss': loss, 'lm_logits': lm_logits}
    self.parent.assertListEqual(list(result['loss'].size()), [])
    self.parent.assertListEqual(list(result['lm_logits'].size()), [self.
        batch_size, self.seq_length, self.vocab_size])