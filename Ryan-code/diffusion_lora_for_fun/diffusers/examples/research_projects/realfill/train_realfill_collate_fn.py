def collate_fn(examples):
    input_ids = [example['prompt_ids'] for example in examples]
    images = [example['images'] for example in examples]
    masks = [example['masks'] for example in examples]
    weightings = [example['weightings'] for example in examples]
    conditioning_images = [example['conditioning_images'] for example in
        examples]
    images = torch.stack(images)
    images = images.to(memory_format=torch.contiguous_format).float()
    masks = torch.stack(masks)
    masks = masks.to(memory_format=torch.contiguous_format).float()
    weightings = torch.stack(weightings)
    weightings = weightings.to(memory_format=torch.contiguous_format).float()
    conditioning_images = torch.stack(conditioning_images)
    conditioning_images = conditioning_images.to(memory_format=torch.
        contiguous_format).float()
    input_ids = torch.cat(input_ids, dim=0)
    batch = {'input_ids': input_ids, 'images': images, 'masks': masks,
        'weightings': weightings, 'conditioning_images': conditioning_images}
    return batch
