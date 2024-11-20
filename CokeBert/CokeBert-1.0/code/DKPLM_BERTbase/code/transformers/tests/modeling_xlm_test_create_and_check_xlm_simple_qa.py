def create_and_check_xlm_simple_qa(self, config, input_ids, token_type_ids,
    input_lengths, sequence_labels, token_labels, is_impossible_labels,
    input_mask):
    model = XLMForQuestionAnsweringSimple(config)
    model.eval()
    outputs = model(input_ids)
    outputs = model(input_ids, start_positions=sequence_labels,
        end_positions=sequence_labels)
    loss, start_logits, end_logits = outputs
    result = {'loss': loss, 'start_logits': start_logits, 'end_logits':
        end_logits}
    self.parent.assertListEqual(list(result['start_logits'].size()), [self.
        batch_size, self.seq_length])
    self.parent.assertListEqual(list(result['end_logits'].size()), [self.
        batch_size, self.seq_length])
    self.check_loss_output(result)
