def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in
        examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    final_dict = {'pixel_values': pixel_values}
    final_dict['input_ids'] = torch.stack([example['input_ids'] for example in
        examples])
    return final_dict
