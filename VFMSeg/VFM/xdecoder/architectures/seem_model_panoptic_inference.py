def panoptic_inference(self, mask_cls, mask_pred):
    scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
    mask_pred = mask_pred.sigmoid()
    keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.
        object_mask_threshold)
    cur_scores = scores[keep]
    cur_classes = labels[keep]
    cur_masks = mask_pred[keep]
    cur_mask_cls = mask_cls[keep]
    cur_mask_cls = cur_mask_cls[:, :-1]
    cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks
    h, w = cur_masks.shape[-2:]
    panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.
        device)
    segments_info = []
    panoptic_seg_logits = torch.zeros((h, w, cur_mask_cls.shape[-1]), dtype
        =torch.half, device=cur_masks.device) - 100.0
    current_segment_id = 0
    if cur_masks.shape[0] == 0:
        return panoptic_seg, segments_info
    else:
        cur_mask_ids = cur_prob_masks.argmax(0)
        stuff_memory_list = {}
        for k in range(cur_classes.shape[0]):
            pred_class = cur_classes[k].item()
            isthing = (pred_class in self.metadata.
                thing_dataset_id_to_contiguous_id.values())
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < self.overlap_threshold:
                    continue
                if not isthing:
                    if int(pred_class) in stuff_memory_list.keys():
                        panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                        panoptic_seg_logits[mask] = cur_mask_cls[k]
                        continue
                    else:
                        stuff_memory_list[int(pred_class)
                            ] = current_segment_id + 1
                current_segment_id += 1
                panoptic_seg[mask] = current_segment_id
                panoptic_seg_logits[mask] = cur_mask_cls[k]
                segments_info.append({'id': current_segment_id, 'isthing':
                    bool(isthing), 'category_id': int(pred_class)})
        return panoptic_seg, segments_info, panoptic_seg_logits
