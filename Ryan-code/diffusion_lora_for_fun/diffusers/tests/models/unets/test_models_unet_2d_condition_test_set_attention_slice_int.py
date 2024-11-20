@require_torch_gpu
def test_set_attention_slice_int(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    unet = self.get_unet_model()
    unet.set_attention_slice(2)
    latents = self.get_latents(33)
    encoder_hidden_states = self.get_encoder_hidden_states(33)
    timestep = 1
    with torch.no_grad():
        _ = unet(latents, timestep=timestep, encoder_hidden_states=
            encoder_hidden_states).sample
    mem_bytes = torch.cuda.max_memory_allocated()
    assert mem_bytes < 5 * 10 ** 9
