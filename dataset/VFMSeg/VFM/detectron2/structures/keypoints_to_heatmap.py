def to_heatmap(self, boxes: torch.Tensor, heatmap_size: int) ->torch.Tensor:
    """
        Convert keypoint annotations to a heatmap of one-hot labels for training,
        as described in :paper:`Mask R-CNN`.

        Arguments:
            boxes: Nx4 tensor, the boxes to draw the keypoints to

        Returns:
            heatmaps:
                A tensor of shape (N, K), each element is integer spatial label
                in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
            valid:
                A tensor of shape (N, K) containing whether each keypoint is in the roi or not.
        """
    return _keypoints_to_heatmap(self.tensor, boxes, heatmap_size)
