def collate_fn(examples):
    pixel_values = torch.stack([example['pixel_values'] for example in
        examples])
    input_ids = torch.stack([example['input_ids'] for example in examples])
    batch = {'pixel_values': pixel_values, 'input_ids': input_ids}
    batch = {k: v.numpy() for k, v in batch.items()}
    return batch
