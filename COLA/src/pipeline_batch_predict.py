def batch_predict(self, input_dataset, batch_size, top_k):
    device = self._device
    data_collator = DataCollatorWithPadding(self._tokenizer, padding=True)
    train_dataloader = DataLoader(input_dataset, shuffle=False, collate_fn=
        data_collator, batch_size=batch_size)
    topk_value_list, topk_indices_list = [], []
    with torch.no_grad():
        for data_batch in tqdm(train_dataloader, 'predicting'):
            cur_batch_size = len(data_batch['input_ids'])
            data_batch.to(device)
            mask_token_index = torch.where(data_batch['input_ids'] == self.
                _tokenizer.mask_token_id)[1]
            token_logits = self._model(**data_batch).logits
            mask_token_logits = token_logits[torch.arange(cur_batch_size),
                mask_token_index, :]
            mask_token_logits = F.softmax(mask_token_logits, dim=1)
            top_k_values, top_k_indices = torch.topk(mask_token_logits, k=
                top_k, dim=1)
            topk_value_list.append(top_k_values.cpu().numpy())
            topk_indices_list.append(top_k_indices.cpu().numpy())
    topk_value_list = np.concatenate(topk_value_list)
    topk_indices_list = np.concatenate(topk_indices_list)
    return topk_value_list, topk_indices_list
