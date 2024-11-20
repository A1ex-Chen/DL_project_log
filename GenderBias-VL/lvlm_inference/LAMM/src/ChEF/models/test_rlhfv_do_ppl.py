def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    tokenized = self.tokenizer(batch_prompt)
    input_ids = [torch.as_tensor(v) for v in tokenized['input_ids']]
    input_ids = torch_pad_sequence(input_ids, self.tokenizer.pad_token_id,
        padding_side='left')
    input_size = input_ids.shape[-1]
    attn_mask = [torch.as_tensor(v) for v in tokenized['attention_mask']]
    attn_mask = torch_pad_sequence(attn_mask, 0, padding_side='left')
    outputs = self.model(input_ids=input_ids.to(self.device), images=
        batch_images, attention_mask=attn_mask.to(self.device), labels=
        input_ids, use_cache=None, output_attentions=None,
        output_hidden_states=None, return_dict=None)
    logits = outputs.logits
    logits = logits[:, :-1].float()
    labels = input_ids
    labels = labels[:, 1:]
    results = []
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.tokenizer.encode(f'{option}',
            return_tensors='pt', add_special_tokens=False).squeeze(0))
    for idx in range(labels.shape[0]):
        option_len = len(batch_option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
        start_index = non_zero_indices.max() - option_len + 1
        end_index = start_index + option_len
        if not np.all(labels[idx][start_index:end_index].detach().cpu().
            numpy() == batch_option_ids[idx].numpy()):
            import ipdb
            ipdb.set_trace()
        prob = F.softmax(logits[idx][start_index:end_index], dim=-1)
        rows = torch.arange(0, option_len)
        score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean(
            ).item()
        results.append(score)
    return results
