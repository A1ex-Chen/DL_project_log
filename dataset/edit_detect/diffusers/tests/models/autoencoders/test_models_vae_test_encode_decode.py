@torch.no_grad()
def test_encode_decode(self):
    vae = ConsistencyDecoderVAE.from_pretrained('openai/consistency-decoder')
    vae.to(torch_device)
    image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/img2img/sketch-mountains-input.jpg'
        ).resize((256, 256))
    image = torch.from_numpy(np.array(image).transpose(2, 0, 1).astype(np.
        float32) / 127.5 - 1)[None, :, :, :].cuda()
    latent = vae.encode(image).latent_dist.mean
    sample = vae.decode(latent, generator=torch.Generator('cpu').manual_seed(0)
        ).sample
    actual_output = sample[0, :2, :2, :2].flatten().cpu()
    expected_output = torch.tensor([-0.0141, -0.0014, 0.0115, 0.0086, 
        0.1051, 0.1053, 0.1031, 0.1024])
    assert torch_all_close(actual_output, expected_output, atol=0.005)
