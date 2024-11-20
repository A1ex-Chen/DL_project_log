@staticmethod
def rgblike_to_depthmap(image: Union[np.ndarray, torch.Tensor]) ->Union[np.
    ndarray, torch.Tensor]:
    """
        Args:
            image: RGB-like depth image

        Returns: depth map

        """
    return image[:, :, 1] * 2 ** 8 + image[:, :, 2]
