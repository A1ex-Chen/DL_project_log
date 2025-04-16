def create_and_check_xlnet_qa(self, config, input_ids_1, input_ids_2,
    input_ids_q, perm_mask, input_mask, target_mapping, segment_ids,
    lm_labels, sequence_labels, is_impossible_labels):
    model = TFXLNetForQuestionAnsweringSimple(config)
    inputs = {'input_ids': input_ids_1, 'attention_mask': input_mask,
        'token_type_ids': segment_ids}
    start_logits, end_logits, mems = model(inputs)
    result = {'start_logits': start_logits.numpy(), 'end_logits':
        end_logits.numpy(), 'mems': [m.numpy() for m in mems]}
    self.parent.assertListEqual(list(result['start_logits'].shape), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['end_logits'].shape), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
