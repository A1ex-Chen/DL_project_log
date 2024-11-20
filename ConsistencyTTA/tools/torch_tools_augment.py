def augment(waveforms, texts, num_items=None):
    """ num_items is the number of augmented examples per batch
    """
    if num_items == None:
        num_items = len(texts) // 2
    if torch.is_tensor(waveforms):
        waveforms = waveforms.cpu().numpy()
    combinations = list(itertools.combinations(list(range(len(texts))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]
    mixed_sound_list, mixed_caption_list = [], []
    for i, j in selected_combinations:
        mixed_sound = mix(waveforms[i, :], waveforms[j, :], 0.5, 16000)
        mixed_caption = f'{texts[i]} and {uncapitalize(texts[j])}'
        mixed_sound_list.append(mixed_sound.reshape(1, -1))
        mixed_caption_list.append(mixed_caption)
    mixed_waveforms = torch.tensor(np.concatenate(mixed_sound_list, 0))
    mixed_waveforms = mixed_waveforms / mixed_waveforms.abs().max() / 2
    return mixed_waveforms, mixed_caption_list
