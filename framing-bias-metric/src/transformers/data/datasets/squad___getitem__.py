def __getitem__(self, i) ->Dict[str, torch.Tensor]:
    feature = self.features[i]
    input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
    attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
    token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
    cls_index = torch.tensor(feature.cls_index, dtype=torch.long)
    p_mask = torch.tensor(feature.p_mask, dtype=torch.float)
    is_impossible = torch.tensor(feature.is_impossible, dtype=torch.float)
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
        'token_type_ids': token_type_ids}
    if self.args.model_type in ['xlm', 'roberta', 'distilbert', 'camembert']:
        del inputs['token_type_ids']
    if self.args.model_type in ['xlnet', 'xlm']:
        inputs.update({'cls_index': cls_index, 'p_mask': p_mask})
        if self.args.version_2_with_negative:
            inputs.update({'is_impossible': is_impossible})
        if self.is_language_sensitive:
            inputs.update({'langs': torch.ones(input_ids.shape, dtype=torch
                .int64) * self.args.lang_id})
    if self.mode == Split.train:
        start_positions = torch.tensor(feature.start_position, dtype=torch.long
            )
        end_positions = torch.tensor(feature.end_position, dtype=torch.long)
        inputs.update({'start_positions': start_positions, 'end_positions':
            end_positions})
    return inputs
