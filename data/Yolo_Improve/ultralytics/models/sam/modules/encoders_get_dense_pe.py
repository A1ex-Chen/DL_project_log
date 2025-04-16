def get_dense_pe(self) ->torch.Tensor:
    """
        Returns the positional encoding used to encode point prompts, applied to a dense set of points the shape of the
        image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape 1x(embed_dim)x(embedding_h)x(embedding_w)
        """
    return self.pe_layer(self.image_embedding_size).unsqueeze(0)
