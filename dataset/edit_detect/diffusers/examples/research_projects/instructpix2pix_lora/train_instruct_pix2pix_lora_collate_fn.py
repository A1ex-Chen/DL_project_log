def collate_fn(examples):
    original_pixel_values = torch.stack([example['original_pixel_values'] for
        example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.
        contiguous_format).float()
    edited_pixel_values = torch.stack([example['edited_pixel_values'] for
        example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.
        contiguous_format).float()
    input_ids = torch.stack([example['input_ids'] for example in examples])
    return {'original_pixel_values': original_pixel_values,
        'edited_pixel_values': edited_pixel_values, 'input_ids': input_ids}
