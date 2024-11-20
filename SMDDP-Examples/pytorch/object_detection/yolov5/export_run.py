@smart_inference_mode()
def run(data=ROOT / 'data/coco128.yaml', weights=ROOT / 'yolov5s.pt', imgsz
    =(640, 640), batch_size=1, device='cpu', include=('torchscript', 'onnx'
    ), half=False, inplace=False, train=False, keras=False, optimize=False,
    int8=False, dynamic=False, simplify=False, opset=12, verbose=False,
    workspace=4, nms=False, agnostic_nms=False, topk_per_class=100,
    topk_all=100, iou_thres=0.45, conf_thres=0.25):
    t = time.time()
    include = [x.lower() for x in include]
    fmts = tuple(export_formats()['Argument'][1:])
    flags = [(x in include) for x in fmts]
    assert sum(flags) == len(include
        ), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    (jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
        ) = flags
    file = Path(url2file(weights) if str(weights).startswith(('http:/',
        'https:/')) else weights)
    device = select_device(device)
    if half:
        assert device.type != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = attempt_load(weights, device=device, inplace=True, fuse=True)
    nc, names = model.nc, model.names
    imgsz *= 2 if len(imgsz) == 1 else 1
    assert nc == len(names
        ), f'Model class count {nc} != len(names) {len(names)}'
    if optimize:
        assert device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'
    gs = int(max(model.stride))
    imgsz = [check_img_size(x, gs) for x in imgsz]
    im = torch.zeros(batch_size, 3, *imgsz).to(device)
    model.train() if train else model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.onnx_dynamic = dynamic
            m.export = True
    for _ in range(2):
        y = model(im)
    if half and not coreml:
        im, model = im.half(), model.half()
    shape = tuple(y[0].shape)
    LOGGER.info(
        f"""
{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)"""
        )
    f = [''] * 10
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
    if jit:
        f[0] = export_torchscript(model, im, file, optimize)
    if engine:
        f[1] = export_engine(model, im, file, train, half, dynamic,
            simplify, workspace, verbose)
    if onnx or xml:
        f[2] = export_onnx(model, im, file, opset, train, dynamic, simplify)
    if xml:
        f[3] = export_openvino(model, file, half)
    if coreml:
        _, f[4] = export_coreml(model, im, file, int8, half)
    if any((saved_model, pb, tflite, edgetpu, tfjs)):
        if int8 or edgetpu:
            check_requirements(('flatbuffers==1.12',))
        assert not tflite or not tfjs, 'TFLite and TF.js models must be exported separately, please pass only one type.'
        model, f[5] = export_saved_model(model.cpu(), im, file, dynamic,
            tf_nms=nms or agnostic_nms or tfjs, agnostic_nms=agnostic_nms or
            tfjs, topk_per_class=topk_per_class, topk_all=topk_all,
            iou_thres=iou_thres, conf_thres=conf_thres, keras=keras)
        if pb or tfjs:
            f[6] = export_pb(model, file)
        if tflite or edgetpu:
            f[7] = export_tflite(model, im, file, int8=int8 or edgetpu,
                data=data, nms=nms, agnostic_nms=agnostic_nms)
        if edgetpu:
            f[8] = export_edgetpu(file)
        if tfjs:
            f[9] = export_tfjs(file)
    f = [str(x) for x in f if x]
    if any(f):
        h = '--half' if half else ''
        LOGGER.info(
            f"""
Export complete ({time.time() - t:.2f}s)
Results saved to {colorstr('bold', file.parent.resolve())}
Detect:          python detect.py --weights {f[-1]} {h}
Validate:        python val.py --weights {f[-1]} {h}
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')
Visualize:       https://netron.app"""
            )
    return f
