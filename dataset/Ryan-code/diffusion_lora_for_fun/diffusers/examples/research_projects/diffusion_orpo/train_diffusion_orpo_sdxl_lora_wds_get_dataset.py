def get_dataset(args):
    dataset = wds.WebDataset(args.dataset_path, resampled=True, handler=wds
        .warn_and_continue).shuffle(690, handler=wds.warn_and_continue).decode(
        'pil', handler=wds.warn_and_continue).rename(original_prompt=
        'original_prompt.txt', jpg_0='jpg_0.jpg', jpg_1='jpg_1.jpg',
        label_0='label_0.txt', label_1='label_1.txt', handler=wds.
        warn_and_continue)
    return dataset
