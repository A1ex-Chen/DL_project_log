def get_audio_embedding(self, data):
    """Get the audio embedding from the model

        Parameters
        ----------
        data: a list of dict
            the audio input dict list from 'get_audio_feature' method

        Returns
        ----------
        audio_embed: torch.Tensor
            a tensor of audio_embeds (N, D)

        """
    device = next(self.parameters()).device
    input_dict = {}
    keys = data[0].keys()
    for k in keys:
        input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(
            device)
    audio_embeds = self.audio_projection(self.encode_audio(input_dict,
        device=device)['embedding'])
    audio_embeds = F.normalize(audio_embeds, dim=-1)
    return audio_embeds
