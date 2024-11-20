def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    self.generator.ppl = True
    batch_images = [img for image_list in batch_images for img in image_list]
    batch_samples = self.make_batches(batch_images, batch_prompt)
    logits = []
    labels = []
    for sample in batch_samples:
        probs = self.task.inference_step(self.generator, [self.model],
            sample, constraints=None)
        sample_logits = probs[:, :-1].float()
        sample_labels = sample['net_input']['src_tokens'][:, 1:]
        logits.append(sample_logits[0])
        labels.append(sample_labels[0])
    results = []
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(get_token_src(self.task, option, self.
            tokenizer, self.special_tokens))
    for idx in range(len(labels)):
        option_len = len(batch_option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx], as_tuple=False).squeeze()
        start_index = non_zero_indices.max() - option_len + 1
        end_index = start_index + option_len
        if not np.all(labels[idx][start_index:end_index].detach().cpu().
            numpy() == np.array(batch_option_ids[idx])):
            import ipdb
            ipdb.set_trace()
        prob = F.softmax(logits[idx][start_index:end_index], dim=-1)
        rows = torch.arange(0, option_len)
        score = torch.log(prob[rows, batch_option_ids[idx][:option_len]]).mean(
            ).item()
        results.append(score)
    return results
