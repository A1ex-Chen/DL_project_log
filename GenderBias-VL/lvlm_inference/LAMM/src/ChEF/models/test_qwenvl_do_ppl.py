def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    raw_text, context_tokens = [item[0] for item in batch_prompt], [torch.
        tensor(item[1]) for item in batch_prompt]
    input_ids = torch.nn.utils.rnn.pad_sequence(context_tokens, batch_first
        =True, padding_value=0.0).to(self.device)
    attention_mask = input_ids.ne(0.0)
    transformer_outputs = self.model.transformer(input_ids=input_ids,
        attention_mask=attention_mask)
    hidden_states = transformer_outputs[0]
    logits = self.model.lm_head(hidden_states)
    logits = logits[:, :-1].float()
    labels = input_ids
    labels = labels[:, 1:]
    results = []
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.tokenizer.encode(f' {option}',
            add_special_tokens=False, return_tensors='pt').squeeze(0))
    for idx in range(labels.shape[0]):
        option_len = len(batch_option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
        start_index = non_zero_indices.max() - option_len + 1
        end_index = start_index + option_len
        last_zero_num = option_len - torch.nonzero(batch_option_ids[idx],
            as_tuple=False).squeeze().max() - 1
        if last_zero_num > 0:
            start_index += last_zero_num
            end_index += last_zero_num
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
