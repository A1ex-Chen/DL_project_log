@smart_inference_mode()
def run(model_name, dataset_name, outfile, data=ROOT / 'data/coco128.yaml',
    imgsz=(640, 640), batch_size=1, device='cpu', include=('torchscript',
    'onnx'), half=False, inplace=False, keras=False, optimize=False, int8=
    False, dynamic=False, simplify=False, opset=12, verbose=False,
    workspace=4, nms=False, agnostic_nms=False, topk_per_class=100,
    topk_all=100, iou_thres=0.45, conf_thres=0.25, no_post_processing=False):
    t_start = time.time()
    file = Path(outfile) if outfile is not None else Path(model_name)
    include = [x.lower() for x in include]
    fmts = tuple(export_formats()['Argument'][1:])
    flags = [(x in include) for x in fmts]
    assert sum(flags) == len(include
        ), f'ERROR: Invalid --include {include}, valid --include arguments are {fmts}'
    (jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs,
        paddle) = flags
    device = select_device(device)
    if half:
        assert device.type != 'cpu' or coreml, '--half only compatible with GPU export, i.e. use --device 0'
        assert not dynamic, '--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both'
    model = get_model(model_name=model_name, dataset_name=dataset_name,
        pretrained=False)
    if not hasattr(model, 'stride'):
        model.stride = torch.tensor([32.0])
    if hasattr(model, 'names') and isinstance(model.names, (list, tuple)):
        model.names = dict(enumerate(model.names))
    model = model.fuse().eval() if hasattr(model, 'fuse') else model.eval()
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU,
            Detect, DetectionModel):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    imgsz *= 2 if len(imgsz) == 1 else 1
    if optimize:
        assert device.type == 'cpu', '--optimize not compatible with cuda devices, i.e. use --device cpu'
    gs = int(max(model.stride))
    imgsz = [check_img_size(x, gs) for x in imgsz]
    im = torch.zeros(batch_size, 3, *imgsz).to(device)
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True
            m.no_post_processing = no_post_processing
    for _ in range(2):
        y = model(im)
    if half and not coreml:
        im, model = im.half(), model.half()
    if not no_post_processing:
        shape = tuple((y[0] if isinstance(y, tuple) else y).shape)
        LOGGER.info(
            f"""
{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)"""
            )
    metadata = {'stride': int(max(model.stride)), 'names': model.names}
    f = [''] * len(fmts)
    warnings.filterwarnings(action='ignore', category=torch.jit.TracerWarning)
    if jit:
        f[0], _ = export_torchscript(model, im, file, optimize)
    if engine:
        f[1], _ = export_engine(model, im, file, half, dynamic, simplify,
            workspace, verbose)
    if onnx or xml:
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify,
            no_post_processing)
    if xml:
        f[3], _ = export_openvino(file, metadata, half)
    if coreml:
        f[4], _ = export_coreml(model, im, file, int8, half)
    if any((saved_model, pb, tflite, edgetpu, tfjs)):
        assert not tflite or not tfjs, 'TFLite and TF.js models must be exported separately, please pass only one type.'
        f[5], s_model = export_saved_model(model.cpu(), im, file, dynamic,
            tf_nms=nms or agnostic_nms or tfjs, agnostic_nms=agnostic_nms or
            tfjs, topk_per_class=topk_per_class, topk_all=topk_all,
            iou_thres=iou_thres, conf_thres=conf_thres, keras=keras)
        if pb or tfjs:
            f[6], _ = export_pb(s_model, file)
        if tflite or edgetpu:
            f[7], _ = export_tflite(s_model, im, file, int8 or edgetpu,
                data=data, nms=nms, agnostic_nms=agnostic_nms)
            if edgetpu:
                f[8], _ = export_edgetpu(file)
            add_tflite_metadata(f[8] or f[7], metadata, num_outputs=len(
                s_model.outputs))
        if tfjs:
            f[9], _ = export_tfjs(file, int8)
    if paddle:
        f[10], _ = export_paddle(model, im, file, metadata)
    f = [str(x) for x in f if x]
    if any(f):
        cls, seg, det = False, False, True
        dir = Path('segment' if seg else 'classify' if cls else '')
        h = '--half' if half else ''
        s = (
            '# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference'
             if cls else 
            '# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference'
             if seg else '')
        LOGGER.info(
            f"""
Export complete ({time.time() - t_start:.1f}s)
Results saved to {colorstr('bold', file.parent.resolve())}
Detect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}
Validate:        python {dir / 'val.py'} --weights {f[-1]} {h}
PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}
Visualize:       https://netron.app"""
            )
    return f
