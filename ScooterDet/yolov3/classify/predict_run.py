@smart_inference_mode()
def run(weights=ROOT / 'yolov5s-cls.pt', source=ROOT / 'data/images', data=
    ROOT / 'data/coco128.yaml', imgsz=(224, 224), device='', view_img=False,
    save_txt=False, nosave=False, augment=False, visualize=False, update=
    False, project=ROOT / 'runs/predict-cls', name='exp', exist_ok=False,
    half=False, dnn=False, vid_stride=1):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in IMG_FORMATS + VID_FORMATS
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://',
        'https://'))
    webcam = source.isnumeric() or source.endswith('.streams'
        ) or is_url and not is_file
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
        exist_ok=True)
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data,
        fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, transforms=
            classify_transforms(imgsz[0]), vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride,
            auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, transforms=
            classify_transforms(imgsz[0]), vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            if len(im.shape) == 3:
                im = im[None]
        with dt[1]:
            results = model(im)
        with dt[2]:
            pred = F.softmax(results, dim=1)
        for i, prob in enumerate(pred):
            seen += 1
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.
                mode == 'image' else f'_{frame}')
            s += '%gx%g ' % im.shape[2:]
            annotator = Annotator(im0, example=str(names), pil=True)
            top5i = prob.argsort(0, descending=True)[:5].tolist()
            s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "
            text = '\n'.join(f'{prob[j]:.2f} {names[j]}' for j in top5i)
            if save_img or view_img:
                annotator.text([32, 32], text, txt_color=(255, 255, 255))
            if save_txt:
                with open(f'{txt_path}.txt', 'a') as f:
                    f.write(text + '\n')
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.
                        WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path[i] != save_path:
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.
                            VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        LOGGER.info(f'{s}{dt[1].dt * 1000.0:.1f}ms')
    t = tuple(x.t / seen * 1000.0 for x in dt)
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {1, 3, *imgsz}'
         % t)
    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}"
             if save_txt else '')
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])
