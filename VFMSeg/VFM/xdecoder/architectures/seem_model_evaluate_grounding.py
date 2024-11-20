def evaluate_grounding(self, batched_inputs, mode):
    images = [x['image'].to(self.device) for x in batched_inputs]
    images = [((x - self.pixel_mean) / self.pixel_std) for x in images]
    images = ImageList.from_tensors(images, self.size_divisibility)
    assert len(images.tensor
        ) == 1, 'grounding evaluation only support single batch size now'
    extra = {}
    mask_pred_results = []
    for idx, batch_per_image in enumerate(batched_inputs):
        grd_texts = batch_per_image['groundings']['texts']
        grd_texts = [x[0] for x in grd_texts]
        gtext = (self.sem_seg_head.predictor.lang_encoder.
            get_text_token_embeddings(grd_texts, name='grounding', token=
            False, norm=False))
        token_emb = gtext['token_emb']
        tokens = gtext['tokens']
        query_emb = token_emb[tokens['attention_mask'].bool()]
        non_zero_query_mask = torch.zeros(query_emb[:, None].shape[:-1],
            dtype=torch.bool, device=query_emb.device)
        extra['grounding_tokens'] = query_emb[:, None]
        extra['grounding_nonzero_mask'] = non_zero_query_mask.t()
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
        mask_pred_results += [pred_gmasks[matched_id, :, :]]
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
