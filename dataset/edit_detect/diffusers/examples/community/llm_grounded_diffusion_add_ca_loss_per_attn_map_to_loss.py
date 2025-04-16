def add_ca_loss_per_attn_map_to_loss(self, loss, attn_map, object_number,
    bboxes, phrase_indices, fg_top_p=0.2, bg_top_p=0.2, fg_weight=1.0,
    bg_weight=1.0):
    b, i, j = attn_map.shape
    H = W = int(math.sqrt(i))
    for obj_idx in range(object_number):
        obj_loss = 0
        mask = torch.zeros(size=(H, W), device='cuda')
        obj_boxes = bboxes[obj_idx]
        if not isinstance(obj_boxes[0], Iterable):
            obj_boxes = [obj_boxes]
        for obj_box in obj_boxes:
            x_min, y_min, x_max, y_max = scale_proportion(obj_box, H=H, W=W)
            mask[y_min:y_max, x_min:x_max] = 1
        for obj_position in phrase_indices[obj_idx]:
            ca_map_obj = attn_map[:, :, obj_position].reshape(b, H, W)
            ca_map_obj = attn_map[:, :, obj_position]
            k_fg = (mask.sum() * fg_top_p).long().clamp_(min=1)
            k_bg = ((1 - mask).sum() * bg_top_p).long().clamp_(min=1)
            mask_1d = mask.view(1, -1)
            obj_loss += (1 - (ca_map_obj * mask_1d).topk(k=k_fg).values.
                mean(dim=1)).sum(dim=0) * fg_weight
            obj_loss += (ca_map_obj * (1 - mask_1d)).topk(k=k_bg).values.mean(
                dim=1).sum(dim=0) * bg_weight
        loss += obj_loss / len(phrase_indices[obj_idx])
    return loss
