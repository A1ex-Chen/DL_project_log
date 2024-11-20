@torch.no_grad()
def convert_models(model_path: str, controlnet_path: list, output_path: str,
    opset: int, fp16: bool=False, sd_xl: bool=False):
    """
    Function to convert models in stable diffusion controlnet pipeline into ONNX format

    Example:
    python convert_stable_diffusion_controlnet_to_onnx.py
    --model_path danbrown/RevAnimated-v1-2-2
    --controlnet_path lllyasviel/control_v11f1e_sd15_tile ioclab/brightness-controlnet
    --output_path path-to-models-stable_diffusion/RevAnimated-v1-2-2
    --fp16

    Example for SD XL:
    python convert_stable_diffusion_controlnet_to_onnx.py
    --model_path stabilityai/stable-diffusion-xl-base-1.0
    --controlnet_path SargeZT/sdxl-controlnet-seg
    --output_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0
    --fp16
    --sd_xl

    Returns:
        create 4 onnx models in output path
        text_encoder/model.onnx
        unet/model.onnx + unet/weights.pb
        vae_encoder/model.onnx
        vae_decoder/model.onnx

        run test script in diffusers/examples/community
        python test_onnx_controlnet.py
        --sd_model danbrown/RevAnimated-v1-2-2
        --onnx_model_dir path-to-models-stable_diffusion/RevAnimated-v1-2-2
        --qr_img_path path-to-qr-code-image
    """
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = 'cuda'
    elif fp16 and not torch.cuda.is_available():
        raise ValueError(
            '`float16` model export is only supported on GPUs with CUDA')
    else:
        device = 'cpu'
    controlnets = []
    for path in controlnet_path:
        controlnet = ControlNetModel.from_pretrained(path, torch_dtype=dtype
            ).to(device)
        if is_torch_2_0_1:
            controlnet.set_attn_processor(AttnProcessor())
        controlnets.append(controlnet)
    if sd_xl:
        if len(controlnets) == 1:
            controlnet = controlnets[0]
        else:
            raise ValueError('MultiControlNet is not yet supported.')
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_path, controlnet=controlnet, torch_dtype=dtype, variant=
            'fp16', use_safetensors=True).to(device)
    else:
        pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path, controlnet=controlnets, torch_dtype=dtype).to(device)
    output_path = Path(output_path)
    if is_torch_2_0_1:
        pipeline.unet.set_attn_processor(AttnProcessor())
        pipeline.vae.set_attn_processor(AttnProcessor())
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer('A sample prompt', padding='max_length',
        max_length=pipeline.tokenizer.model_max_length, truncation=True,
        return_tensors='pt')
    onnx_export(pipeline.text_encoder, model_args=text_input.input_ids.to(
        device=device, dtype=torch.int32), output_path=output_path /
        'text_encoder' / 'model.onnx', ordered_input_names=['input_ids'],
        output_names=['last_hidden_state', 'pooler_output'], dynamic_axes={
        'input_ids': {(0): 'batch', (1): 'sequence'}}, opset=opset)
    del pipeline.text_encoder
    if sd_xl:
        controlnets = torch.nn.ModuleList(controlnets)
        unet_controlnet = UNet2DConditionXLControlNetModel(pipeline.unet,
            controlnets)
        unet_in_channels = pipeline.unet.config.in_channels
        unet_sample_size = pipeline.unet.config.sample_size
        text_hidden_size = 2048
        img_size = 8 * unet_sample_size
        unet_path = output_path / 'unet' / 'model.onnx'
        onnx_export(unet_controlnet, model_args=(torch.randn(2,
            unet_in_channels, unet_sample_size, unet_sample_size).to(device
            =device, dtype=dtype), torch.tensor([1.0]).to(device=device,
            dtype=dtype), torch.randn(2, num_tokens, text_hidden_size).to(
            device=device, dtype=dtype), torch.randn(len(controlnets), 2, 3,
            img_size, img_size).to(device=device, dtype=dtype), torch.randn
            (len(controlnets), 1).to(device=device, dtype=dtype), torch.
            randn(2, 1280).to(device=device, dtype=dtype), torch.rand(2, 6)
            .to(device=device, dtype=dtype)), output_path=unet_path,
            ordered_input_names=['sample', 'timestep',
            'encoder_hidden_states', 'controlnet_conds',
            'conditioning_scales', 'text_embeds', 'time_ids'], output_names
            =['noise_pred'], dynamic_axes={'sample': {(0): '2B', (2): 'H',
            (3): 'W'}, 'encoder_hidden_states': {(0): '2B'},
            'controlnet_conds': {(1): '2B', (3): '8H', (4): '8W'},
            'text_embeds': {(0): '2B'}, 'time_ids': {(0): '2B'}}, opset=
            opset, use_external_data_format=True)
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        shape_inference.infer_shapes_path(unet_model_path, unet_model_path)
        unet_opt_graph = optimize(onnx.load(unet_model_path), name='Unet',
            verbose=True)
        shutil.rmtree(unet_dir)
        os.mkdir(unet_dir)
        onnx.save_model(unet_opt_graph, unet_model_path,
            save_as_external_data=True, all_tensors_to_one_file=True,
            location='weights.pb', convert_attribute=False)
        del pipeline.unet
    else:
        controlnets = torch.nn.ModuleList(controlnets)
        unet_controlnet = UNet2DConditionControlNetModel(pipeline.unet,
            controlnets)
        unet_in_channels = pipeline.unet.config.in_channels
        unet_sample_size = pipeline.unet.config.sample_size
        img_size = 8 * unet_sample_size
        unet_path = output_path / 'unet' / 'model.onnx'
        onnx_export(unet_controlnet, model_args=(torch.randn(2,
            unet_in_channels, unet_sample_size, unet_sample_size).to(device
            =device, dtype=dtype), torch.tensor([1.0]).to(device=device,
            dtype=dtype), torch.randn(2, num_tokens, text_hidden_size).to(
            device=device, dtype=dtype), torch.randn(len(controlnets), 2, 3,
            img_size, img_size).to(device=device, dtype=dtype), torch.randn
            (len(controlnets), 1).to(device=device, dtype=dtype)),
            output_path=unet_path, ordered_input_names=['sample',
            'timestep', 'encoder_hidden_states', 'controlnet_conds',
            'conditioning_scales'], output_names=['noise_pred'],
            dynamic_axes={'sample': {(0): '2B', (2): 'H', (3): 'W'},
            'encoder_hidden_states': {(0): '2B'}, 'controlnet_conds': {(1):
            '2B', (3): '8H', (4): '8W'}}, opset=opset,
            use_external_data_format=True)
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        shape_inference.infer_shapes_path(unet_model_path, unet_model_path)
        unet_opt_graph = optimize(onnx.load(unet_model_path), name='Unet',
            verbose=True)
        shutil.rmtree(unet_dir)
        os.mkdir(unet_dir)
        onnx.save_model(unet_opt_graph, unet_model_path,
            save_as_external_data=True, all_tensors_to_one_file=True,
            location='weights.pb', convert_attribute=False)
        del pipeline.unet
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    vae_encoder.forward = lambda sample: vae_encoder.encode(sample
        ).latent_dist.sample()
    onnx_export(vae_encoder, model_args=(torch.randn(1, vae_in_channels,
        vae_sample_size, vae_sample_size).to(device=device, dtype=dtype),),
        output_path=output_path / 'vae_encoder' / 'model.onnx',
        ordered_input_names=['sample'], output_names=['latent_sample'],
        dynamic_axes={'sample': {(0): 'batch', (1): 'channels', (2):
        'height', (3): 'width'}}, opset=opset)
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    vae_decoder.forward = vae_encoder.decode
    onnx_export(vae_decoder, model_args=(torch.randn(1, vae_latent_channels,
        unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
        ), output_path=output_path / 'vae_decoder' / 'model.onnx',
        ordered_input_names=['latent_sample'], output_names=['sample'],
        dynamic_axes={'latent_sample': {(0): 'batch', (1): 'channels', (2):
        'height', (3): 'width'}}, opset=opset)
    del pipeline.vae
    del pipeline
