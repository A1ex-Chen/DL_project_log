def _compare(self, boxes1, boxes2):
    """Compute matrix of (negated) sq distances.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.

    Returns:
      A tensor with shape [N, M] representing negated pairwise squared distance.
    """
    return box_np_ops.distance_similarity(boxes1[..., [0, 1, -1]], boxes2[
        ..., [0, 1, -1]], dist_norm=self._distance_norm, with_rotation=self
        ._with_rotation, rot_alpha=self._rotation_alpha)
