def collate_fn(examples):
    input_ids = [example['instance_prompt_ids'] for example in examples]
    pixel_values = [example['instance_images'] for example in examples]
    if args.with_prior_preservation:
        input_ids += [example['class_prompt_ids'] for example in examples]
        pixel_values += [example['class_images'] for example in examples]
        pior_pil = [example['class_PIL_images'] for example in examples]
    masks = []
    masked_images = []
    for example in examples:
        pil_image = example['PIL_images']
        mask = random_mask(pil_image.size, 1, False)
        mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
        masks.append(mask)
        masked_images.append(masked_image)
    if args.with_prior_preservation:
        for pil_image in pior_pil:
            mask = random_mask(pil_image.size, 1, False)
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
            masks.append(mask)
            masked_images.append(masked_image)
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format
        ).float()
    input_ids = tokenizer.pad({'input_ids': input_ids}, padding=True,
        return_tensors='pt').input_ids
    masks = torch.stack(masks)
    masked_images = torch.stack(masked_images)
    batch = {'input_ids': input_ids, 'pixel_values': pixel_values, 'masks':
        masks, 'masked_images': masked_images}
    return batch
