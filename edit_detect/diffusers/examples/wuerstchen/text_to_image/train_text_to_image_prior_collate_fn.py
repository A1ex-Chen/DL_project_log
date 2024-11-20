def collate_fn(examples):
    effnet_pixel_values = torch.stack([example['effnet_pixel_values'] for
        example in examples])
    effnet_pixel_values = effnet_pixel_values.to(memory_format=torch.
        contiguous_format).float()
    text_input_ids = torch.stack([example['text_input_ids'] for example in
        examples])
    text_mask = torch.stack([example['text_mask'] for example in examples])
    return {'effnet_pixel_values': effnet_pixel_values, 'text_input_ids':
        text_input_ids, 'text_mask': text_mask}
