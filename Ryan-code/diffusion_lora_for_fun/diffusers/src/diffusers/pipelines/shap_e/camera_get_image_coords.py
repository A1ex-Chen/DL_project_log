def get_image_coords(self) ->torch.Tensor:
    """
        :return: coords of shape (width * height, 2)
        """
    pixel_indices = torch.arange(self.height * self.width)
    coords = torch.stack([pixel_indices % self.width, torch.div(
        pixel_indices, self.width, rounding_mode='trunc')], axis=1)
    return coords
