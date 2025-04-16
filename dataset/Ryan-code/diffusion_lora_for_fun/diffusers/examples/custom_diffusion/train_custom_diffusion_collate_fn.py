def collate_fn(examples, with_prior_preservation):
    input_ids = [example['instance_prompt_ids'] for example in examples]
    pixel_values = [example['instance_images'] for example in examples]
    mask = [example['mask'] for example in examples]
    if with_prior_preservation:
        input_ids += [example['class_prompt_ids'] for example in examples]
        pixel_values += [example['class_images'] for example in examples]
        mask += [example['class_mask'] for example in examples]
    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()
    batch = {'input_ids': input_ids, 'pixel_values': pixel_values, 'mask':
        mask.unsqueeze(1)}
    return batch
