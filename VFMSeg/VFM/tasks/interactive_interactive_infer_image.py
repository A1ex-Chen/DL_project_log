def interactive_infer_image(model, audio_model, image, tasks, refimg=None,
    reftxt=None, audio_pth=None, video_pth=None):
    image_ori = transform(image['image'])
    mask_ori = image['mask']
    width = image_ori.size[0]
    height = image_ori.size[1]
    image_ori = np.asarray(image_ori)
    visual = Visualizer(image_ori, metadata=metadata)
    images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
    data = {'image': images, 'height': height, 'width': width}
    if len(tasks) == 0:
        tasks = ['Panoptic']
    model.model.task_switch['spatial'] = False
    model.model.task_switch['visual'] = False
    model.model.task_switch['grounding'] = False
    model.model.task_switch['audio'] = False
    example = None
    if 'Example' in tasks:
        model.model.task_switch['visual'] = True
        model.model.task_switch['spatial'] = True
        refimg_ori, refimg_mask = refimg['image'], refimg['mask']
        refimg_ori = transform(refimg_ori)
        _width = refimg_ori.size[0]
        _height = refimg_ori.size[1]
        refimg_ori = np.asarray(refimg_ori)
        refimg_ori_np = refimg_ori.copy()
        images = torch.from_numpy(refimg_ori.copy()).permute(2, 0, 1).cuda()
        batched_inputs = [{'image': images, 'height': _height, 'width':
            _width, 'spatial_query': {}}]
        refimg_mask = np.asarray(refimg_mask)[:, :, 0:1].copy()
        refimg_mask = torch.from_numpy(refimg_mask).permute(2, 0, 1)[None,]
        refimg_mask = F.interpolate(refimg_mask, (_height, _width), mode=
            'bilinear') > 0
        batched_inputs[0]['spatial_query']['rand_shape'] = refimg_mask
        outputs_refimg, img_shape = model.model.evaluate_referring_image(
            batched_inputs)
        model.model.task_switch['spatial'] = False
        data['visual'] = outputs_refimg
    stroke = None
    if 'Stroke' in tasks:
        model.model.task_switch['spatial'] = True
        mask_ori = np.asarray(mask_ori)[:, :, 0:1].copy()
        mask_ori = torch.from_numpy(mask_ori).permute(2, 0, 1)[None,]
        mask_ori = F.interpolate(mask_ori, (height, width), mode='bilinear'
            ) > 0
        data['stroke'] = mask_ori
    text = None
    if 'Text' in tasks:
        model.model.task_switch['grounding'] = True
        data['text'] = [reftxt]
    audio = None
    if 'Audio' in tasks:
        model.model.task_switch['audio'] = True
        audio_result = audio_model.transcribe(audio_pth)
        data['audio'] = [audio_result['text']]
    batch_inputs = [data]
    if 'Panoptic' in tasks:
        model.model.metadata = metadata
        results = model.model.evaluate(batch_inputs)
        pano_seg = results[-1]['panoptic_seg'][0]
        pano_seg_info = results[-1]['panoptic_seg'][1]
        demo = visual.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info)
        res = demo.get_image()
        return Image.fromarray(res), None
    else:
        results, image_size, extra = model.model.evaluate_demo(batch_inputs)
    if 'Stroke' in tasks:
        v_emb = results['pred_maskembs']
        s_emb = results['pred_pspatials']
        pred_masks = results['pred_masks']
        pred_logits = v_emb @ s_emb.transpose(1, 2)
        logits_idx_y = pred_logits[:, :, 0].max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.
            device)
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        pred_masks_pos = pred_masks[logits_idx]
        pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]
    elif 'Example' in tasks:
        v_emb = results['pred_maskembs']
        s_emb = results['pred_pvisuals']
        pred_masks = results['pred_masks']
        pred_logits = v_emb @ s_emb.transpose(1, 2)
        logits_idx_y = pred_logits[:, :, 0].max(dim=1)[1]
        logits_idx_x = torch.arange(len(logits_idx_y), device=logits_idx_y.
            device)
        logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
        pred_masks_pos = pred_masks[logits_idx]
        pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]
    elif 'Text' in tasks:
        pred_masks = results['pred_masks'][0]
        v_emb = results['pred_captions'][0]
        t_emb = extra['grounding_class']
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-07)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-07)
        temperature = (model.model.sem_seg_head.predictor.lang_encoder.
            logit_scale)
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        matched_id = out_prob.max(0)[1]
        pred_masks_pos = pred_masks[matched_id, :, :]
        pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]
    elif 'Audio' in tasks:
        pred_masks = results['pred_masks'][0]
        v_emb = results['pred_captions'][0]
        t_emb = extra['audio_class']
        t_emb = t_emb / (t_emb.norm(dim=-1, keepdim=True) + 1e-07)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-07)
        temperature = (model.model.sem_seg_head.predictor.lang_encoder.
            logit_scale)
        out_prob = vl_similarity(v_emb, t_emb, temperature=temperature)
        matched_id = out_prob.max(0)[1]
        pred_masks_pos = pred_masks[matched_id, :, :]
        pred_class = results['pred_logits'][0][matched_id].max(dim=-1)[1]
    pred_masks_pos = (F.interpolate(pred_masks_pos[None,], image_size[-2:],
        mode='bilinear')[0, :, :data['height'], :data['width']] > 0.0).float(
        ).cpu().numpy()
    texts = [all_classes[pred_class[0]]]
    for idx, mask in enumerate(pred_masks_pos):
        out_txt = texts[idx] if 'Text' not in tasks else reftxt
        demo = visual.draw_binary_mask(mask, color=colors_list[pred_class[0
            ] % 133], text=out_txt)
    res = demo.get_image()
    torch.cuda.empty_cache()
    return Image.fromarray(res), None
