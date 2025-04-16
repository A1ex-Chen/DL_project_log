def __init__(self, keypoints: Union[torch.Tensor, np.ndarray, List[List[
    float]]]):
    """
        Arguments:
            keypoints: A Tensor, numpy array, or list of the x, y, and visibility of each keypoint.
                The shape should be (N, K, 3) where N is the number of
                instances, and K is the number of keypoints per instance.
        """
    device = keypoints.device if isinstance(keypoints, torch.Tensor
        ) else torch.device('cpu')
    keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
    assert keypoints.dim() == 3 and keypoints.shape[2] == 3, keypoints.shape
    self.tensor = keypoints
