def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in
        examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    clip_pixel_values = torch.stack([example['clip_pixel_values'] for
        example in examples])
    clip_pixel_values = clip_pixel_values.to(memory_format=torch.
        contiguous_format).float()
    return {'pixel_values': pixel_values, 'clip_pixel_values':
        clip_pixel_values}
