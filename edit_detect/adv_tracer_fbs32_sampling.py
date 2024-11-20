def sampling(pipeline, num: int, path: str, num_inference_steps: int=1,
    prompts: List=None, prompts_arg_name: str=None, batch_size: int=32,
    generator: torch.Generator=None):
    if prompts is None or len(prompts) == 0:
        used_prompts: List = [None for i in range(num)]
    else:
        used_prompts: List = [prompts[i % len(prompts)] for i in range(num)]
    generator = set_generator(generator=generator)
    os.makedirs(path, exist_ok=True)
    for batch_idx, prompt in enumerate(batchify(data=used_prompts,
        batch_size=batch_size)):
        batch_size: int = len(prompt)
        if prompts is None:
            prompt = None
        if prompts_arg_name is not None:
            images = pipeline(batch_size=batch_size, num_inference_steps=
                num_inference_steps, generator=generator, **{
                prompts_arg_name: prompt}).images
        elif isinstance(prompt, int):
            images = pipeline(batch_size=batch_size, num_inference_steps=
                num_inference_steps, generator=generator, class_labels=prompt
                ).images
        elif isinstance(prompt, str):
            images = pipeline(batch_size=batch_size, num_inference_steps=
                num_inference_steps, generator=generator, prompt=prompt).images
        else:
            images = pipeline(batch_size=batch_size, num_inference_steps=
                num_inference_steps, generator=generator, prompt=prompt).images
        for idx, image in enumerate(images):
            image.save(os.path.join(path,
                f'{batch_idx * batch_size + idx}.jpg'))
