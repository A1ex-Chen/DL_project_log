def do_ppl(self, batch_images, ds_list, batch_options, **kwargs):
    text, images = [], []
    for ds in ds_list:
        model_inputs = ds.to_model_input()
        text.append(model_inputs['input_text'])
        images.append(model_inputs['images'].to(dtype=torch.float16, device
            =self.device))
    images = torch.cat(images, dim=0)
    input_dict = self.tokenizer(text, padding='longest', return_length=True,
        add_special_tokens=False, return_tensors='pt').to(self.device)
    option_dict = self.tokenizer(batch_options, padding='longest',
        return_length=True, add_special_tokens=False, return_tensors='pt')
    input_ids = input_dict['input_ids']
    attention_mask = input_dict['attention_mask']
    option_ids = option_dict['input_ids']
    outputs = self.model.model(input_ids=input_ids, attention_mask=
        attention_mask, images=images)
    hidden_states = outputs[0]
    logits = self.model.lm_head(hidden_states)
    logits = logits[:, :-1].float()
    labels = input_ids[:, 1:]
    results = []
    for idx in range(labels.shape[0]):
        option_len = torch.sum(option_ids[idx] != 0).item()
        end_index = len(labels[idx]) - 1
        start_index = end_index - option_len
        if not np.all(labels[idx][start_index:end_index].detach().cpu().
            numpy() == option_ids[idx][-option_len:].numpy()):
            import ipdb
            ipdb.set_trace()
        prob = F.softmax(logits[idx][start_index:end_index], dim=-1)
        rows = torch.arange(0, option_len)
        score = torch.log(prob[rows, option_ids[idx][-option_len:]]).mean(
            ).item()
        results.append(score)
    return results
