def create_and_check_xlnet_sequence_classif(self, config, input_ids_1,
    input_ids_2, input_ids_q, perm_mask, input_mask, target_mapping,
    segment_ids, lm_labels, sequence_labels, is_impossible_labels):
    model = XLNetForSequenceClassification(config)
    model.eval()
    logits, mems_1 = model(input_ids_1)
    loss, logits, mems_1 = model(input_ids_1, labels=sequence_labels)
    result = {'loss': loss, 'mems_1': mems_1, 'logits': logits}
    self.parent.assertListEqual(list(result['loss'].size()), [])
    self.parent.assertListEqual(list(result['logits'].size()), [self.
        batch_size, self.type_sequence_label_size])
    self.parent.assertListEqual(list(list(mem.size()) for mem in result[
        'mems_1']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
