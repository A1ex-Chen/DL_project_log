def do_ppl(self, batch_images, batch_prompt, batch_options, **kwargs):
    batch_answers = kwargs['batch_answers']
    batch_images = torch.stack(batch_images, dim=0).to(self.device)
    batch_option_ids = []
    for option in batch_options:
        batch_option_ids.append(self.tokenizer.encode(f'{option}',
            add_special_tokens=False, return_tensors='pt').squeeze(0))
    output, labels = self.model.forward_multiple({'image': batch_images,
        'text_input': batch_prompt, 'text_output': batch_answers})
    logits = output['logits'][:, :-1].float()
    labels = labels[:, 1:]
    results = []
    for idx in range(labels.shape[0]):
        option_len = len(batch_option_ids[idx])
        non_zero_indices = torch.nonzero(labels[idx] != -100, as_tuple=False
            ).squeeze()
        start_index = non_zero_indices.max() - option_len
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
