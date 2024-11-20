@require_torch_gpu
def test_stable_cascade_unet_single_file_prior_forward_pass(self):
    dtype = torch.bfloat16
    generator = torch.Generator('cpu')
    model_inputs = {'sample': randn_tensor((1, 16, 24, 24), generator=
        generator.manual_seed(0)).to('cuda', dtype), 'timestep_ratio':
        torch.tensor([1]).to('cuda', dtype), 'clip_text_pooled':
        randn_tensor((1, 1, 1280), generator=generator.manual_seed(0)).to(
        'cuda', dtype), 'clip_text': randn_tensor((1, 77, 1280), generator=
        generator.manual_seed(0)).to('cuda', dtype), 'clip_img':
        randn_tensor((1, 1, 768), generator=generator.manual_seed(0)).to(
        'cuda', dtype), 'pixels': randn_tensor((1, 3, 8, 8), generator=
        generator.manual_seed(0)).to('cuda', dtype)}
    unet = StableCascadeUNet.from_pretrained('stabilityai/stable-cascade-prior'
        , subfolder='prior', revision='refs/pr/2', variant='bf16',
        torch_dtype=dtype)
    unet.to('cuda')
    with torch.no_grad():
        prior_output = unet(**model_inputs).sample.float().cpu().numpy()
    del unet
    gc.collect()
    torch.cuda.empty_cache()
    single_file_url = (
        'https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_c_bf16.safetensors'
        )
    single_file_unet = StableCascadeUNet.from_single_file(single_file_url,
        torch_dtype=dtype)
    single_file_unet.to('cuda')
    with torch.no_grad():
        prior_single_file_output = single_file_unet(**model_inputs
            ).sample.float().cpu().numpy()
    del single_file_unet
    gc.collect()
    torch.cuda.empty_cache()
    max_diff = numpy_cosine_similarity_distance(prior_output.flatten(),
        prior_single_file_output.flatten())
    assert max_diff < 0.008
