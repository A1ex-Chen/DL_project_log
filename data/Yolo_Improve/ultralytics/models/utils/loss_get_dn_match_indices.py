@staticmethod
def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
    """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
    dn_match_indices = []
    idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
    for i, num_gt in enumerate(gt_groups):
        if num_gt > 0:
            gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
            gt_idx = gt_idx.repeat(dn_num_group)
            assert len(dn_pos_idx[i]) == len(gt_idx
                ), 'Expected the same length, '
            f"""but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively."""
            dn_match_indices.append((dn_pos_idx[i], gt_idx))
        else:
            dn_match_indices.append((torch.zeros([0], dtype=torch.long),
                torch.zeros([0], dtype=torch.long)))
    return dn_match_indices
