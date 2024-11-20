def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example['instance_images'] for example in examples]
    prompts = [example['instance_prompt'] for example in examples]
    original_sizes = [example['original_size'] for example in examples]
    crop_top_lefts = [example['crop_top_left'] for example in examples]
    if with_prior_preservation:
        pixel_values += [example['class_images'] for example in examples]
        prompts += [example['class_prompt'] for example in examples]
        original_sizes += [example['original_size'] for example in examples]
        crop_top_lefts += [example['crop_top_left'] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    batch = {'pixel_values': pixel_values, 'prompts': prompts,
        'original_sizes': original_sizes, 'crop_top_lefts': crop_top_lefts}
    return batch
