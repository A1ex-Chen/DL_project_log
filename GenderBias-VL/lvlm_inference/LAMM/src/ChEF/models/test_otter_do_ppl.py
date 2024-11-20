def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    vision_x = torch.stack(batch_images, dim=0).to(self.model.device, dtype
        =self.dtype)
    lang_x = self.model.text_tokenizer(batch_prompt, return_tensors='pt',
        padding=True)
    input_ids = lang_x['input_ids'].to(self.model.device)
    attention_mask = lang_x['attention_mask'].to(self.model.device, dtype=
        self.dtype)
    output = self.model(vision_x=vision_x, lang_x=input_ids, attention_mask
        =attention_mask)
    logits = output['logits'][:, :-1].float()
    labels = input_ids[:, 1:]
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.model.text_tokenizer(option,
            add_special_tokens=False, return_tensors='pt')['input_ids'].
            squeeze(0))
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
