def interactive_infer_video(model, audio_model, image, tasks, refimg=None,
    reftxt=None, audio_pth=None, video_pth=None):
    if 'Video' in tasks:
        input_dir = video_pth.replace('.mp4', '')
        input_name = input_dir.split('/')[-1]
        random_number = str(random.randint(10000, 99999))
        output_dir = input_dir + '_output'
        output_name = output_dir.split('/')[-1]
        output_file = video_pth.replace('.mp4', '_{}_output.mp4'.format(
            random_number))
        frame_interval = 10
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ffmpeg_cmd = 'ffmpeg -i {} -vf "fps=5" {}/%04d.png'.format(video_pth,
            input_dir)
        os.system(ffmpeg_cmd)
        data = {}
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
        model.model.task_switch['visual'] = False
        model.model.task_switch['spatial'] = False
        data['visual'] = outputs_refimg
        model.model.task_switch['visual'] = True
        frame_pths = sorted(glob.glob(os.path.join(input_dir, '*.png')))
        for frame_pth in frame_pths:
            image_ori = transform(Image.open(frame_pth))
            width = image_ori.size[0]
            height = image_ori.size[1]
            image_ori = np.asarray(image_ori)
            visual = Visualizer(image_ori[:, :, ::-1], metadata=metadata)
            images = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).cuda()
            data.update({'image': images, 'height': height, 'width': width})
            batch_inputs = [data]
            results, image_size, extra = model.model.evaluate_demo(batch_inputs
                )
            v_emb = results['pred_maskembs']
            s_emb = results['pred_pvisuals']
            pred_masks = results['pred_masks']
            pred_logits = v_emb @ s_emb.transpose(1, 2)
            logits_idx_y = pred_logits[:, :, 0].max(dim=1)[1]
            logits_idx_x = torch.arange(len(logits_idx_y), device=
                logits_idx_y.device)
            logits_idx = torch.stack([logits_idx_x, logits_idx_y]).tolist()
            pred_masks_pos = pred_masks[logits_idx]
            pred_class = results['pred_logits'][logits_idx].max(dim=-1)[1]
            pred_masks_pos = (F.interpolate(pred_masks_pos[None,],
                image_size[-2:], mode='bilinear')[0, :, :data['height'], :
                data['width']] > 0.0).float().cpu().numpy()
            texts = [all_classes[pred_class[0]]]
            for idx, mask in enumerate(pred_masks_pos):
                out_txt = texts[idx]
                demo = visual.draw_binary_mask(mask, color=colors_list[
                    pred_class[0] % 133], text=out_txt)
            res = demo.get_image()
            output_pth = frame_pth.replace(input_name, output_name)
            cv2.imwrite(output_pth, res)
        ffmpeg_cmd = (
            "ffmpeg -framerate 5 -pattern_type glob -i '{}/*.png' -c:v libx264  {}"
            .format(output_dir, output_file))
        os.system(ffmpeg_cmd)
        return None, output_file
