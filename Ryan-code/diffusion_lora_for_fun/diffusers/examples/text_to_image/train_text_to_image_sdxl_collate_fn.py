def collate_fn(examples):
    model_input = torch.stack([torch.tensor(example['model_input']) for
        example in examples])
    original_sizes = [example['original_sizes'] for example in examples]
    crop_top_lefts = [example['crop_top_lefts'] for example in examples]
    prompt_embeds = torch.stack([torch.tensor(example['prompt_embeds']) for
        example in examples])
    pooled_prompt_embeds = torch.stack([torch.tensor(example[
        'pooled_prompt_embeds']) for example in examples])
    return {'model_input': model_input, 'prompt_embeds': prompt_embeds,
        'pooled_prompt_embeds': pooled_prompt_embeds, 'original_sizes':
        original_sizes, 'crop_top_lefts': crop_top_lefts}
