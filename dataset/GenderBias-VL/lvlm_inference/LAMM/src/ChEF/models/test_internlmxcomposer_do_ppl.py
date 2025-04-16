def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    results = []
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.tokenizer.encode(f' {option}',
            add_special_tokens=False, return_tensors='pt').squeeze(0))
    to_regress_embeds, attention_mask, targets, im_mask, token_ids = (self.
        model.interleav_wrap(batch_images, batch_prompt))
    im_mask = im_mask.bool()
    outputs = self.model.model(input_ids=None, attention_mask=
        attention_mask, position_ids=None, past_key_values=None,
        inputs_embeds=to_regress_embeds, im_mask=im_mask)
    hidden_states = outputs[0]
    logits = self.model.output(hidden_states)
    logits = logits.float()
    logits = logits[:, :-1]
    labels = token_ids[:, 1:]
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
