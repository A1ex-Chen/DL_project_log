@require_torch_accelerator
def test_from_pretrained_accelerate_wont_change_results(self):
    model_accelerate, _ = UNet2DModel.from_pretrained(
        'fusing/unet-ldm-dummy-update', output_loading_info=True)
    model_accelerate.to(torch_device)
    model_accelerate.eval()
    noise = torch.randn(1, model_accelerate.config.in_channels,
        model_accelerate.config.sample_size, model_accelerate.config.
        sample_size, generator=torch.manual_seed(0))
    noise = noise.to(torch_device)
    time_step = torch.tensor([10] * noise.shape[0]).to(torch_device)
    arr_accelerate = model_accelerate(noise, time_step)['sample']
    del model_accelerate
    torch.cuda.empty_cache()
    gc.collect()
    model_normal_load, _ = UNet2DModel.from_pretrained(
        'fusing/unet-ldm-dummy-update', output_loading_info=True,
        low_cpu_mem_usage=False)
    model_normal_load.to(torch_device)
    model_normal_load.eval()
    arr_normal_load = model_normal_load(noise, time_step)['sample']
    assert torch_all_close(arr_accelerate, arr_normal_load, rtol=0.001)
