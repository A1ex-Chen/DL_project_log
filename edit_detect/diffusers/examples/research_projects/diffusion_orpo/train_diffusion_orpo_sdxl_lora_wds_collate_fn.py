def collate_fn(samples):
    pixel_values = torch.stack([sample['pixel_values'] for sample in samples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    original_sizes = [example['original_size'] for example in samples]
    crop_top_lefts = [example['crop_top_left'] for example in samples]
    input_ids_one = torch.stack([example['tokens_one'] for example in samples])
    input_ids_two = torch.stack([example['tokens_two'] for example in samples])
    return {'pixel_values': pixel_values, 'input_ids_one': input_ids_one,
        'input_ids_two': input_ids_two, 'original_sizes': original_sizes,
        'crop_top_lefts': crop_top_lefts}
