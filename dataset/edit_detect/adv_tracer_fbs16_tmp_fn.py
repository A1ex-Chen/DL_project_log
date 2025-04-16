def tmp_fn():
    model_type: str = ModelSched.MD_CLASS_CM
    model_id: str = ModelSched.MD_NAME_CD_LPIPS_IMAGENET64
    num_classes: int = 1000
    sample_num: int = 3000
    path: str = os.path.join('fake_images', model_id)
    num_inference_steps: int = 1
    prompts: List[int] = [i for i in range(num_classes)]
    prompts_arg_name: str = 'class_labels'
    batch_size: int = 32
    device: Union[str, torch.device] = 'cuda:1'
    seed: int = 0
    g = torch.Generator(device=device).manual_seed(seed)
    pipe, unet, vae, scheduler = ModelSched.get_model_sched(model_type=
        model_type, model_id=model_id)
    pipe, unet, vae, scheduler = ModelSched.all_to_device(pipe, unet, vae,
        scheduler, device=device)
    pipe.unet, unet, vae = ModelSched.all_compile(pipe.unet, unet, vae)
    sampling(pipeline=pipe, num=sample_num, path=path, num_inference_steps=
        num_inference_steps, prompts=prompts, prompts_arg_name=
        prompts_arg_name, batch_size=batch_size, generator=g)
