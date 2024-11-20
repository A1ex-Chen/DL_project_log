def __call__(self, ft_root, vname):
    all_ft = []
    for ft_name in self.visual_ft:
        ft_path = os.path.join(ft_root, ft_name, vname)
        all_ft.append(np.load(ft_path + '.npy'))
    for ft_name in self.audio_ft:
        ft_path = os.path.join(ft_root, ft_name, vname)
        all_ft.append(np.load(ft_path + '.npy'))
    min_len = min([len(ft) for ft in all_ft])
    sampled_ft = [ft[:min_len] for ft in all_ft]
    sampled_ft = np.concatenate(sampled_ft, axis=1)
    item = {}
    item['video_fts'] = torch.Tensor(sampled_ft)
    video_type_token = self.tokenizer.convert_tokens_to_ids('<video>')
    item['token_type_ids'] = torch.Tensor([video_type_token] * len(sampled_ft)
        ).long()
    return item
