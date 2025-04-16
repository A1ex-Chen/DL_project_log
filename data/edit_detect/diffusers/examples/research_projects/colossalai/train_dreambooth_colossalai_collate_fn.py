def collate_fn(examples):
    input_ids = [example['instance_prompt_ids'] for example in examples]
    pixel_values = [example['instance_images'] for example in examples]
    if args.with_prior_preservation:
        input_ids += [example['class_prompt_ids'] for example in examples]
        pixel_values += [example['class_images'] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    input_ids = tokenizer.pad({'input_ids': input_ids}, padding=
        'max_length', max_length=tokenizer.model_max_length, return_tensors
        ='pt').input_ids
    batch = {'input_ids': input_ids, 'pixel_values': pixel_values}
    return batch
