@staticmethod
def pt_to_numpy(images):
    """
        Convert a numpy image to a pytorch tensor
        """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images
