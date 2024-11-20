def collate_fn(num_instances, examples, with_prior_preservation=False):
    input_ids = []
    pixel_values = []
    for i in range(num_instances):
        input_ids += [example[f'instance_prompt_ids_{i}'] for example in
            examples]
        pixel_values += [example[f'instance_images_{i}'] for example in
            examples]
    if with_prior_preservation:
        for i in range(num_instances):
            input_ids += [example[f'class_prompt_ids_{i}'] for example in
                examples]
            pixel_values += [example[f'class_images_{i}'] for example in
                examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    input_ids = torch.cat(input_ids, dim=0)
    batch = {'input_ids': input_ids, 'pixel_values': pixel_values}
    return batch
