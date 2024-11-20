def main(args: argparse.Namespace):
    image_path = args.img
    net_h, net_w = args.img_size
    if not args.show and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.load_param(args.param)
    net.load_model(args.bin)
    ex = net.create_extractor()
    img = cv2.imread(image_path)
    draw_img = img.copy()
    img_w = img.shape[1]
    img_h = img.shape[0]
    w = img_w
    h = img_h
    scale = 1.0
    if w > h:
        scale = float(net_w) / w
        w = net_w
        h = int(h * scale)
    else:
        scale = float(net_h) / h
        h = net_h
        w = int(w * scale)
    mat_in = ncnn.Mat.from_pixels_resize(img, ncnn.Mat.PixelType.
        PIXEL_BGR2RGB, img_w, img_h, w, h)
    wpad = (w + args.max_stride - 1) // args.max_stride * args.max_stride - w
    hpad = (h + args.max_stride - 1) // args.max_stride * args.max_stride - h
    mat_in_pad = ncnn.copy_make_border(mat_in, hpad // 2, hpad - hpad // 2,
        wpad // 2, wpad - wpad // 2, ncnn.BorderType.BORDER_CONSTANT, 114.0)
    mat_in_pad.substract_mean_normalize([0, 0, 0], [1 / 225, 1 / 225, 1 / 225])
    ex.input('in0', mat_in_pad)
    ret1, mat_out1 = ex.extract('out0')
    ret2, mat_out2 = ex.extract('out1')
    ret3, mat_out3 = ex.extract('out2')
    if args.max_stride == 64:
        ret4, mat_out4 = ex.extract('out3')
    outputs = [np.array(mat_out1), np.array(mat_out2), np.array(mat_out3)]
    if args.max_stride == 64:
        outputs.append(np.array(mat_out4))
    nmsd_boxes, nmsd_scores, nmsd_labels = yolov6_decode(outputs,
        CONF_THRES, IOU_THRES)
    for box, score, label in zip(nmsd_boxes, nmsd_scores, nmsd_labels):
        x0, y0, x1, y1 = box
        x0 = x0 - wpad / 2
        y0 = y0 - hpad / 2
        x1 = x1 - wpad / 2
        y1 = y1 - hpad / 2
        name = CLASS_NAMES[label]
        box_color = CLASS_COLORS[label]
        x0 = math.floor(min(max(x0 / scale, 1), img_w - 1))
        y0 = math.floor(min(max(y0 / scale, 1), img_h - 1))
        x1 = math.ceil(min(max(x1 / scale, 1), img_w - 1))
        y1 = math.ceil(min(max(y1 / scale, 1), img_h - 1))
        cv2.rectangle(draw_img, (x0, y0), (x1, y1), box_color, 2)
        cv2.putText(draw_img, f'{name}: {score:.2f}', (x0, max(y0 - 5, 1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    if args.show:
        cv2.imshow('res', draw_img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(image_path)
            ), draw_img)
