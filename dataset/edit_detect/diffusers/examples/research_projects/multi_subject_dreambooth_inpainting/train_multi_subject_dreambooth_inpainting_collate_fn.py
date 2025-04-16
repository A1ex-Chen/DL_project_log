def collate_fn(examples, tokenizer):
    input_ids = [example['instance_prompt_id'] for example in examples]
    pixel_values = [example['instance_image'] for example in examples]
    masks, masked_images = [], []
    for example in examples:
        mask = weighted_mask(example['instance_masks'])
        mask, masked_image = prepare_mask_and_masked_image(example[
            'PIL_image'], mask)
        masks.append(mask)
        masked_images.append(masked_image)
    pixel_values = torch.stack(pixel_values).to(memory_format=torch.
        contiguous_format).float()
    masks = torch.stack(masks)
    masked_images = torch.stack(masked_images)
    input_ids = tokenizer.pad({'input_ids': input_ids}, padding=True,
        return_tensors='pt').input_ids
    batch = {'input_ids': input_ids, 'pixel_values': pixel_values, 'masks':
        masks, 'masked_images': masked_images}
    return batch
