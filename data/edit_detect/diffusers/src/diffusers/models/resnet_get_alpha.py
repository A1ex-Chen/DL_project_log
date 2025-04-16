def get_alpha(self, image_only_indicator: torch.Tensor, ndims: int
    ) ->torch.Tensor:
    if self.merge_strategy == 'fixed':
        alpha = self.mix_factor
    elif self.merge_strategy == 'learned':
        alpha = torch.sigmoid(self.mix_factor)
    elif self.merge_strategy == 'learned_with_images':
        if image_only_indicator is None:
            raise ValueError(
                'Please provide image_only_indicator to use learned_with_images merge strategy'
                )
        alpha = torch.where(image_only_indicator.bool(), torch.ones(1, 1,
            device=image_only_indicator.device), torch.sigmoid(self.
            mix_factor)[..., None])
        if ndims == 5:
            alpha = alpha[:, None, :, None, None]
        elif ndims == 3:
            alpha = alpha.reshape(-1)[:, None, None]
        else:
            raise ValueError(
                f'Unexpected ndims {ndims}. Dimensions should be 3 or 5')
    else:
        raise NotImplementedError
    return alpha
