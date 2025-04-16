def generate_proposals_batch(self, anchor_bboxes_batch: List[Tensor],
    anchor_objectnesses_batch: Tensor, anchor_transformers_batch: Tensor,
    padded_image_width: int, padded_image_height: int) ->Tuple[List[Tensor],
    List[Tensor]]:
    batch_size = len(anchor_bboxes_batch)
    pre_nms_top_n = (self._train_pre_nms_top_n if self.training else self.
        _eval_pre_nms_top_n)
    post_nms_top_n = (self._train_post_nms_top_n if self.training else self
        ._eval_post_nms_top_n)
    proposal_bboxes_batch, proposal_probs_batch = [], []
    for b in range(batch_size):
        proposal_bboxes = BBox.apply_transformer(anchor_bboxes_batch[b],
            anchor_transformers_batch[b])
        proposal_bboxes = BBox.clip(proposal_bboxes, left=0, top=0, right=
            padded_image_width, bottom=padded_image_height)
        proposal_probs = F.softmax(input=anchor_objectnesses_batch[b, :, 1],
            dim=-1)
        _, sorted_indices = torch.sort(proposal_probs, dim=-1, descending=True)
        proposal_bboxes = proposal_bboxes[sorted_indices][:pre_nms_top_n]
        proposal_probs = proposal_probs[sorted_indices][:pre_nms_top_n]
        kept_indices = nms(proposal_bboxes, proposal_probs, iou_threshold=
            self._proposal_nms_threshold)
        proposal_bboxes = proposal_bboxes[kept_indices][:post_nms_top_n]
        proposal_probs = proposal_probs[kept_indices][:post_nms_top_n]
        proposal_bboxes_batch.append(proposal_bboxes.detach())
        proposal_probs_batch.append(proposal_probs.detach())
    return proposal_bboxes_batch, proposal_probs_batch
