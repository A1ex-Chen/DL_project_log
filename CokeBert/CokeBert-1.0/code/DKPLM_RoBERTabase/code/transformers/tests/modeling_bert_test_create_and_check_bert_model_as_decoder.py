def create_and_check_bert_model_as_decoder(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels,
    choice_labels, encoder_hidden_states, encoder_attention_mask):
    model = BertModel(config)
    model.eval()
    sequence_output, pooled_output = model(input_ids, attention_mask=
        input_mask, token_type_ids=token_type_ids, encoder_hidden_states=
        encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
    sequence_output, pooled_output = model(input_ids, attention_mask=
        input_mask, token_type_ids=token_type_ids, encoder_hidden_states=
        encoder_hidden_states)
    sequence_output, pooled_output = model(input_ids, attention_mask=
        input_mask, token_type_ids=token_type_ids)
    result = {'sequence_output': sequence_output, 'pooled_output':
        pooled_output}
    self.parent.assertListEqual(list(result['sequence_output'].size()), [
        self.batch_size, self.seq_length, self.hidden_size])
    self.parent.assertListEqual(list(result['pooled_output'].size()), [self
        .batch_size, self.hidden_size])
