def load_dpr_model(self):
    model = DPRReader(DPRConfig(**BertConfig.get_config_dict(
        'bert-base-uncased')[0]))
    print('Loading DPR reader from {}'.format(self.src_file))
    saved_state = load_states_from_checkpoint(self.src_file)
    state_dict = {'encoder.bert_model.embeddings.position_ids': model.
        span_predictor.encoder.bert_model.embeddings.position_ids}
    for key, value in saved_state.model_dict.items():
        if key.startswith('encoder.') and not key.startswith(
            'encoder.encode_proj'):
            key = 'encoder.bert_model.' + key[len('encoder.'):]
        state_dict[key] = value
    model.span_predictor.load_state_dict(state_dict)
    return model
