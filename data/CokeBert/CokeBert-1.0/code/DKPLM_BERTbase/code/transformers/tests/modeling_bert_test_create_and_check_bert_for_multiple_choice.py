def create_and_check_bert_for_multiple_choice(self, config, input_ids,
    token_type_ids, input_mask, sequence_labels, token_labels, choice_labels):
    config.num_choices = self.num_choices
    model = BertForMultipleChoice(config=config)
    model.eval()
    multiple_choice_inputs_ids = input_ids.unsqueeze(1).expand(-1, self.
        num_choices, -1).contiguous()
    multiple_choice_token_type_ids = token_type_ids.unsqueeze(1).expand(-1,
        self.num_choices, -1).contiguous()
    multiple_choice_input_mask = input_mask.unsqueeze(1).expand(-1, self.
        num_choices, -1).contiguous()
    loss, logits = model(multiple_choice_inputs_ids, attention_mask=
        multiple_choice_input_mask, token_type_ids=
        multiple_choice_token_type_ids, labels=choice_labels)
    result = {'loss': loss, 'logits': logits}
    self.parent.assertListEqual(list(result['logits'].size()), [self.
        batch_size, self.num_choices])
    self.check_loss_output(result)
