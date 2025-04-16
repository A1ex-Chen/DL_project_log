def do_ppl(self, batch_images, conversations, batch_options, **kwargs):
    option_ids = []
    for option in batch_options:
        option_token = self.model.llama_tokenizer.encode(option,
            add_special_tokens=False, return_tensors='pt').squeeze(0)
        option_ids.append(option_token)
    logits, labels = self.model.ppl_forward(dict(vision_type='image',
        task_type=self.task_type, vision_paths=batch_images, output_texts=
        conversations))
    logits = logits[:, :-1].float()
    labels = labels[:, 1:]
    results = []
    for idx in range(labels.shape[0]):
        option_len = len(option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx] != -100, as_tuple=False
            ).squeeze()
        end_index = non_zero_indices.max() - 2
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
