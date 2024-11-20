def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in
        examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    conditioning_pixel_values = torch.stack([example[
        'conditioning_pixel_values'] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=
        torch.contiguous_format).float()
    prompt_ids = torch.stack([torch.tensor(example['prompt_embeds']) for
        example in examples])
    add_text_embeds = torch.stack([torch.tensor(example['text_embeds']) for
        example in examples])
    add_time_ids = torch.stack([torch.tensor(example['time_ids']) for
        example in examples])
    return {'pixel_values': pixel_values, 'conditioning_pixel_values':
        conditioning_pixel_values, 'prompt_ids': prompt_ids,
        'unet_added_conditions': {'text_embeds': add_text_embeds,
        'time_ids': add_time_ids}}
