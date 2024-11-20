def create_and_check_xlnet_qa(self, config, input_ids_1, input_ids_2,
    input_ids_q, perm_mask, input_mask, target_mapping, segment_ids,
    lm_labels, sequence_labels, is_impossible_labels):
    model = XLNetForQuestionAnswering(config)
    model.eval()
    outputs = model(input_ids_1)
    (start_top_log_probs, start_top_index, end_top_log_probs, end_top_index,
        cls_logits, mems) = outputs
    outputs = model(input_ids_1, start_positions=sequence_labels,
        end_positions=sequence_labels, cls_index=sequence_labels,
        is_impossible=is_impossible_labels, p_mask=input_mask)
    outputs = model(input_ids_1, start_positions=sequence_labels,
        end_positions=sequence_labels, cls_index=sequence_labels,
        is_impossible=is_impossible_labels)
    total_loss, mems = outputs
    outputs = model(input_ids_1, start_positions=sequence_labels,
        end_positions=sequence_labels)
    total_loss, mems = outputs
    result = {'loss': total_loss, 'start_top_log_probs':
        start_top_log_probs, 'start_top_index': start_top_index,
        'end_top_log_probs': end_top_log_probs, 'end_top_index':
        end_top_index, 'cls_logits': cls_logits, 'mems': mems}
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
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
