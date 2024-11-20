def evaluate_grounding_sptial(self, batched_inputs, mode):
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.size_divisibility)
    assert len(images.tensor
        ) == 1, 'grounding evaluation only support single batch size now'
    extra = {}
    dilation = 3
    pos_masks = batched_inputs[0]['spatial_query']['rand_shape'].to(self.device
        ).unbind(0)
    pos_masks = ImageList.from_tensors(pos_masks, self.size_divisibility
        ).tensor
    pos_masks = (F.conv2d(pos_masks.float(), self.dilation_kernel, padding=
        dilation // 2) > 0).unbind(0)
    neg_masks = (batched_inputs[0]['spatial_query']['rand_shape'].to(self.
        device) & False).unbind(0)
    neg_masks = ImageList.from_tensors(neg_masks, self.size_divisibility
        ).tensor.unbind(0)
    mask_pred_results = []
    for idx, batch_per_image in enumerate(batched_inputs):
        grd_texts = batch_per_image['groundings']['texts']
        grd_masks = []
        for idx2, anno_text in enumerate(grd_texts):
            extra.update({'spatial_query_pos_mask': [pos_masks[idx2]],
                'spatial_query_neg_mask': [neg_masks[idx2]]})
            gtext = (self.sem_seg_head.predictor.lang_encoder.
                get_text_token_embeddings([anno_text[0]], name='grounding',
                token=False, norm=False))
            token_emb = gtext['token_emb']
            tokens = gtext['tokens']
            grd_emb = token_emb[0][tokens['attention_mask'].bool()[0]]
            non_zero_query_mask = torch.zeros(grd_emb[:, None].shape[:-1],
                dtype=torch.bool, device=grd_emb.device)
            extra['grounding_tokens'] = grd_emb[:, None]
            extra['grounding_nonzero_mask'] = non_zero_query_mask.t()
            assert len(images.tensor
                ) == 1, 'grounding evaluation only support single batch size now'
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features, extra=extra, task=
                'grounding_eval')
            pred_gmasks = outputs['pred_gmasks'][idx]
            v_emb = outputs['pred_gtexts'][idx]
            t_emb = gtext['class_emb']
            t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-07)
            v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-07)
            temperature = self.sem_seg_head.predictor.lang_encoder.logit_scale
            out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
            matched_id = out_prob.max(0)[1]
            grd_masks += [pred_gmasks[matched_id, :, :]]
        mask_pred_results += [torch.cat(grd_masks)]
    for i in range(len(mask_pred_results)):
        mask_pred_results[i] = F.interpolate(mask_pred_results[i][None,],
            size=(images.tensor.shape[-2], images.tensor.shape[-1]), mode=
            'bilinear', align_corners=False)[0]
    processed_results = []
    for mask_pred_result, input_per_image, image_size in zip(mask_pred_results,
        batched_inputs, images.image_sizes):
        height = input_per_image.get('height', image_size[0])
        width = input_per_image.get('width', image_size[1])
        processed_results.append({})
        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
            mask_pred_result, image_size, height, width)
        processed_results[-1]['grounding_mask'] = mask_pred_result
    return processed_results
