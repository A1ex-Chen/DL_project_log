def generate_results(processor, imgs_dir, visual_dir, jpgs, conf_thres,
    iou_thres, batch_size=1, img_size=[640, 640], shrink_size=0):
    """Run detection on each jpg and write results to file."""
    results = []
    pbar = tqdm(range(math.ceil(len(jpgs) / batch_size)), desc=
        'TRT-Model test in val datasets.')
    idx = 0
    num_visualized = 0
    for _ in pbar:
        imgs = torch.randn((batch_size, 3, 640, 640), dtype=torch.float32,
            device=torch.device('cuda:0'))
        source_imgs = []
        image_names = []
        shapes = []
        for i in range(batch_size):
            if idx == len(jpgs):
                break
            img = cv2.imread(os.path.join(imgs_dir, jpgs[idx]))
            img_src = img.copy()
            h0, w0 = img.shape[:2]
            r = (max(img_size) - shrink_size) / max(h0, w0)
            if r != 1:
                img = cv2.resize(img, (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                    )
            h, w = img.shape[:2]
            imgs[i], pad = processor.pre_process(img)
            source_imgs.append(img_src)
            shape = (h0, w0), ((h / h0, w / w0), pad)
            shapes.append(shape)
            image_names.append(jpgs[idx])
            idx += 1
        output = processor.inference(imgs)
        for j in range(len(shapes)):
            pred = processor.post_process(output[j].unsqueeze(0), shapes[j],
                conf_thres=conf_thres, iou_thres=iou_thres)
            image = source_imgs[j]
            for p in pred:
                x = float(p[0])
                y = float(p[1])
                w = float(p[2] - p[0])
                h = float(p[3] - p[1])
                s = float(p[4])
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y +
                    h)), (255, 0, 0), 1)
            cv2.imwrite('{}'.format(os.path.join(visual_dir, image_names[j]
                )), image)
