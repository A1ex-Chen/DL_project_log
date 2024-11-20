def decode(self, encodings_and_masks, input_tokens, noise_time):
    timesteps = noise_time
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor([timesteps], dtype=torch.long, device=
            input_tokens.device)
    elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(input_tokens.device)
    timesteps = timesteps * torch.ones(input_tokens.shape[0], dtype=
        timesteps.dtype, device=timesteps.device)
    logits = self.decoder(encodings_and_masks=encodings_and_masks,
        decoder_input_tokens=input_tokens, decoder_noise_time=timesteps)
    return logits
