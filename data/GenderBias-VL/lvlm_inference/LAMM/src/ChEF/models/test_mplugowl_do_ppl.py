def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    inputs = self.processor(text=batch_prompt)
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    batch_images = torch.cat(batch_images, dim=0).to(self.device, dtype=
        self.dtype)
    inputs['pixel_values'] = batch_images
    labels = inputs['input_ids'].clone()[:, 1:]
    outputs = self.model.generate(**inputs, ppl=True)
    logits = outputs['logits'][:, :-1].float()
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.tokenizer.encode(f'{option}',
            add_special_tokens=False, return_tensors='pt').squeeze(0))
    results = []
    for idx in range(labels.shape[0]):
        option_len = len(batch_option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx] != -100, as_tuple=False
            ).squeeze()
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
