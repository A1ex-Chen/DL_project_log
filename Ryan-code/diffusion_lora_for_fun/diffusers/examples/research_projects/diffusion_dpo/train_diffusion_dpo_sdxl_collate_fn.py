def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in
        examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    original_sizes = [example['original_sizes'] for example in examples]
    crop_top_lefts = [example['crop_top_lefts'] for example in examples]
    input_ids_one = torch.stack([example['input_ids_one'] for example in
        examples])
    input_ids_two = torch.stack([example['input_ids_two'] for example in
        examples])
    return {'pixel_values': pixel_values, 'input_ids_one': input_ids_one,
        'input_ids_two': input_ids_two, 'original_sizes': original_sizes,
        'crop_top_lefts': crop_top_lefts}
