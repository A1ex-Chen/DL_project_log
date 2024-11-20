def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
    """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
    pred_bboxes, pred_scores = preds
    total_loss = super().forward(pred_bboxes, pred_scores, batch)
    if dn_meta is not None:
        dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta[
            'dn_num_group']
        assert len(batch['gt_groups']) == len(dn_pos_idx)
        match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group,
            batch['gt_groups'])
        dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix=
            '_dn', match_indices=match_indices)
        total_loss.update(dn_loss)
    else:
        total_loss.update({f'{k}_dn': torch.tensor(0.0, device=self.device) for
            k in total_loss.keys()})
    return total_loss
