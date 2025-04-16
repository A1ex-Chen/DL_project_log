def get_text_embedding(self, data):
    """Get the text embedding from the model

        Parameters
        ----------
        data: torch.Tensor 
            a tensor of text embedding

        Returns
        ----------
        text_embed: torch.Tensor
            a tensor of text_embeds (N, D)

        """
    device = next(self.parameters()).device
    for k in data:
        data[k] = data[k].to(device)
    text_embeds = self.encode_text(data, device=device)
    text_embeds = F.normalize(text_embeds, dim=-1)
    return text_embeds
