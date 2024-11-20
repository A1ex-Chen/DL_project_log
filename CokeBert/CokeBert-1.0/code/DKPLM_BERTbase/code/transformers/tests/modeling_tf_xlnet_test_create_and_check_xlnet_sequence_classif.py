def create_and_check_xlnet_sequence_classif(self, config, input_ids_1,
    input_ids_2, input_ids_q, perm_mask, input_mask, target_mapping,
    segment_ids, lm_labels, sequence_labels, is_impossible_labels):
    model = TFXLNetForSequenceClassification(config)
    logits, mems_1 = model(input_ids_1)
    result = {'mems_1': [mem.numpy() for mem in mems_1], 'logits': logits.
        numpy()}
    self.parent.assertListEqual(list(result['logits'].shape), [self.
        batch_size, self.type_sequence_label_size])
    self.parent.assertListEqual(list(list(mem.shape) for mem in result[
        'mems_1']), [[self.seq_length, self.batch_size, self.hidden_size]] *
        self.num_hidden_layers)
