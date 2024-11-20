def create_and_check_xlm_qa(self, config, input_ids, token_type_ids,
    input_lengths, sequence_labels, token_labels, is_impossible_labels,
    input_mask):
    model = XLMForQuestionAnswering(config)
    model.eval()
    outputs = model(input_ids)
    (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index,
        cls_logits) = outputs
    outputs = model(input_ids, start_positions=sequence_labels,
        end_positions=sequence_labels, cls_index=sequence_labels,
        is_impossible=is_impossible_labels, p_mask=input_mask)
    outputs = model(input_ids, start_positions=sequence_labels,
        end_positions=sequence_labels, cls_index=sequence_labels,
        is_impossible=is_impossible_labels)
    total_loss, = outputs
    outputs = model(input_ids, start_positions=sequence_labels,
        end_positions=sequence_labels)
    total_loss, = outputs
    result = {'loss': total_loss, 'start_top_log_probs':
        start_top_log_probs, 'start_top_index': start_top_index,
        'end_top_log_probs': end_top_log_probs, 'end_top_index':
        end_top_index, 'cls_logits': cls_logits}
    self.parent.assertListEqual(list(result['loss'].size()), [])
    self.parent.assertListEqual(list(result['start_top_log_probs'].size()),
        [self.batch_size, model.config.start_n_top])
    self.parent.assertListEqual(list(result['start_top_index'].size()), [
        self.batch_size, model.config.start_n_top])
    self.parent.assertListEqual(list(result['end_top_log_probs'].size()), [
        self.batch_size, model.config.start_n_top * model.config.end_n_top])
    self.parent.assertListEqual(list(result['end_top_index'].size()), [self
        .batch_size, model.config.start_n_top * model.config.end_n_top])
    self.parent.assertListEqual(list(result['cls_logits'].size()), [self.
        batch_size])
