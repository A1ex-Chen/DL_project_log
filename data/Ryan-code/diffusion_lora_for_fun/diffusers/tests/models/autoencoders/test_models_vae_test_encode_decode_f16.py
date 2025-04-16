def test_encode_decode_f16(self):
    vae = ConsistencyDecoderVAE.from_pretrained('openai/consistency-decoder',
        torch_dtype=torch.float16)
    vae.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
        ).resize((256, 256))
    image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.
        float32) / 127.5 - 1)[None, :, :, :].half().cuda()
    latent = vae.encode(image).latent_dist.mean
    sample = vae.decode(latent, generator=torch.Generator('cpu').manual_seed(0)
        ).sample
    actual_output = sample[0, :2, :2, :2].flatten().cpu()
    expected_output = torch.tensor([-0.0111, -0.0125, -0.0017, -0.0007, 
        0.1257, 0.1465, 0.145, 0.1471], dtype=torch.float16)
    assert torch_all_close(actual_output, expected_output, atol=0.005)
