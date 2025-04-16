def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det,
    save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=True):
    """ Model Inference and results visualization """
    vid_path, vid_writer, windows = None, None, []
    fps_calculator = CalcFPS()
    for img_src, img_path, vid_cap in tqdm(self.files):
        img, img_src = self.process_image(img_src, self.img_size, self.
            stride, self.half)
        img = img.to(self.device)
        if len(img.shape) == 3:
            img = img[None]
        t1 = time.time()
        pred_results = self.model(img)
        det = non_max_suppression(pred_results, conf_thres, iou_thres,
            classes, agnostic_nms, max_det=max_det)[0]
        t2 = time.time()
        if self.webcam:
            save_path = osp.join(save_dir, self.webcam_addr)
            txt_path = osp.join(save_dir, self.webcam_addr)
        else:
            rel_path = osp.relpath(osp.dirname(img_path), osp.dirname(self.
                source))
            save_path = osp.join(save_dir, rel_path, osp.basename(img_path))
            txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(
                osp.basename(img_path))[0])
            os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)
        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]
        img_ori = img_src.copy()
        assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
        self.font_check()
        if len(det):
            det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape
                ).round()
            for *xyxy, conf, cls in reversed(det):
                if save_txt:
                    xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) /
                        gn).view(-1).tolist()
                    line = cls, *xywh, conf
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                if save_img:
                    class_num = int(cls)
                    label = (None if hide_labels else self.class_names[
                        class_num] if hide_conf else
                        f'{self.class_names[class_num]} {conf:.2f}')
                    self.plot_box_and_label(img_ori, max(round(sum(img_ori.
                        shape) / 2 * 0.003), 2), xyxy, label, color=self.
                        generate_colors(class_num, True))
            img_src = np.asarray(img_ori)
        fps_calculator.update(1.0 / (t2 - t1))
        avg_fps = fps_calculator.accumulate()
        if self.files.type == 'video':
            self.draw_text(img_src, f'FPS: {avg_fps:0.1f}', pos=(20, 20),
                font_scale=1.0, text_color=(204, 85, 17), text_color_bg=(
                255, 255, 255), font_thickness=2)
        if view_img:
            if img_path not in windows:
                windows.append(img_path)
                cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.
                    WINDOW_KEEPRATIO)
                cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.
                    shape[0])
            cv2.imshow(str(img_path), img_src)
            cv2.waitKey(1)
        if save_img:
            if self.files.type == 'image':
                cv2.imwrite(save_path, img_src)
            else:
                if vid_path != save_path:
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))
                    vid_writer = cv2.VideoWriter(save_path, cv2.
                        VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(img_src)
