def collate_fn(examples):
    original_pixel_values = torch.stack([example['original_pixel_values'] for
        example in examples])
    original_pixel_values = original_pixel_values.to(memory_format=torch.
        contiguous_format).float()
    edited_pixel_values = torch.stack([example['edited_pixel_values'] for
        example in examples])
    edited_pixel_values = edited_pixel_values.to(memory_format=torch.
        contiguous_format).float()
    prompt_embeds = torch.concat([example['prompt_embeds'] for example in
        examples], dim=0)
    add_text_embeds = torch.concat([example['add_text_embeds'] for example in
        examples], dim=0)
    return {'original_pixel_values': original_pixel_values,
        'edited_pixel_values': edited_pixel_values, 'prompt_embeds':
        prompt_embeds, 'add_text_embeds': add_text_embeds}
