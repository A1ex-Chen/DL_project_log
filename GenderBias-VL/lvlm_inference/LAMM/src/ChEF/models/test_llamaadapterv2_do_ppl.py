def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    prompts = [item[0] for item in batch_prompt]
    batch_answers = [item[1] for item in batch_prompt]
    images = torch.stack(batch_images, dim=0).to(self.device)
    logits, labels = self.model.ppl_generate(images, prompts, batch_answers,
        device=self.device)
    logits = logits.float()
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.tokenizer.encode(option, bos=False,
            eos=False))
    results = []
    for idx in range(labels.shape[0]):
        option_len = len(batch_option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx] != -100, as_tuple=False
            ).squeeze()
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
