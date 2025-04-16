@staticmethod
def cat(keypoints_list: List['Keypoints']) ->'Keypoints':
    """
        Concatenates a list of Keypoints into a single Keypoints

        Arguments:
            keypoints_list (list[Keypoints])

        Returns:
            Keypoints: the concatenated Keypoints
        """
    assert isinstance(keypoints_list, (list, tuple))
    assert len(keypoints_list) > 0
    assert all(isinstance(keypoints, Keypoints) for keypoints in keypoints_list
        )
    cat_kpts = type(keypoints_list[0])(torch.cat([kpts.tensor for kpts in
        keypoints_list], dim=0))
    return cat_kpts
