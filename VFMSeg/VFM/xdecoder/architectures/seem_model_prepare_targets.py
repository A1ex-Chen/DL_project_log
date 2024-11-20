def prepare_targets(self, batched_inputs, images):
    h_pad, w_pad = images.tensor.shape[-2:]
    new_targets = []
    for idx, batch_per_image in enumerate(batched_inputs):
        targets_per_image = batch_per_image['instances'].to(self.device)
        gt_masks = targets_per_image.gt_masks.tensor
        padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype
            =gt_masks.dtype, device=gt_masks.device)
        padded_masks[:, :gt_masks.shape[1], :gt_masks.shape[2]] = gt_masks
        gt_boxes = targets_per_image.gt_boxes.tensor
        ratio = torch.tensor([w_pad, h_pad, w_pad, h_pad]).to(gt_boxes.device)[
            None, :]
        gt_boxes = gt_boxes / ratio
        xc, yc, w, h = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2, (gt_boxes[:, 
            1] + gt_boxes[:, 3]) / 2, gt_boxes[:, 2] - gt_boxes[:, 0
            ], gt_boxes[:, 3] - gt_boxes[:, 1]
        gt_boxes = torch.stack([xc, yc, w, h]).permute(1, 0)
        target_dict = {'labels': targets_per_image.gt_classes, 'is_things':
            targets_per_image.is_things, 'masks': padded_masks, 'boxes':
            gt_boxes}
        if self.task_switch['spatial']:
            target_dict['gt_spatial_masks'] = batch_per_image['spatial_query'][
                'gt_masks']
        if self.task_switch['grounding']:
            grd_masks = batch_per_image['groundings']['masks']
            grd_texts = batch_per_image['groundings']['texts']
            grd_hash = batch_per_image['groundings']['hash']
            grd_task = batch_per_image['groundings']['mode']
            if len(grd_masks) == 0:
                padded_masks = None
            else:
                padded_masks = torch.zeros((grd_masks.shape[0], h_pad,
                    w_pad), dtype=grd_masks.dtype, device=grd_masks.device)
                padded_masks[:, :grd_masks.shape[1], :grd_masks.shape[2]
                    ] = grd_masks
            gtext = (self.sem_seg_head.predictor.lang_encoder.
                get_text_token_embeddings(grd_texts, name='grounding',
                token=False, norm=False))
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            unique_hash_id = np.unique(grd_hash, return_index=True)[1]
            selected_mask = np.zeros(len(grd_hash)).astype(np.bool)
            selected_mask[unique_hash_id] = True
            selected_token_emb = token_emb[selected_mask]
            selected_attn_mask = tokens['attention_mask'][selected_mask]
            query_emb = selected_token_emb[selected_attn_mask.bool()]
            class_idx = tokens['attention_mask'].sum(dim=-1) - 1
            class_idx = torch.stack((torch.arange(len(class_idx), device=
                class_idx.device), class_idx)).tolist()
            class_emb = token_emb[class_idx]
            target_dict['grounding_masks'] = padded_masks
            target_dict['grounding_query_embs'] = query_emb
            target_dict['grounding_class_embs'] = class_emb
            target_dict['grounding_hash'] = grd_hash
            target_dict['grounding_task'] = grd_task
        new_targets.append(target_dict)
    return new_targets
