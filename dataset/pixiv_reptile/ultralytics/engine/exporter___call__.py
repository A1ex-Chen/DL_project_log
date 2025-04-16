@smart_inference_mode()
def __call__(self, model=None) ->str:
    """Returns list of exported files/dirs after running callbacks."""
    self.run_callbacks('on_export_start')
    t = time.time()
    fmt = self.args.format.lower()
    if fmt in {'tensorrt', 'trt'}:
        fmt = 'engine'
    if fmt in {'mlmodel', 'mlpackage', 'mlprogram', 'apple', 'ios', 'coreml'}:
        fmt = 'coreml'
    fmts = tuple(export_formats()['Argument'][1:])
    flags = [(x == fmt) for x in fmts]
    if sum(flags) != 1:
        raise ValueError(
            f"Invalid export format='{fmt}'. Valid formats are {fmts}")
    (jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs,
        paddle, ncnn) = flags
    is_tf_format = any((saved_model, pb, tflite, edgetpu, tfjs))
    if fmt == 'engine' and self.args.device is None:
        LOGGER.warning(
            'WARNING ⚠️ TensorRT requires GPU export, automatically assigning device=0'
            )
        self.args.device = '0'
    self.device = select_device('cpu' if self.args.device is None else self
        .args.device)
    if not hasattr(model, 'names'):
        model.names = default_class_names()
    model.names = check_class_names(model.names)
    if self.args.half and self.args.int8:
        LOGGER.warning(
            'WARNING ⚠️ half=True and int8=True are mutually exclusive, setting half=False.'
            )
        self.args.half = False
    if self.args.half and onnx and self.device.type == 'cpu':
        LOGGER.warning(
            'WARNING ⚠️ half=True only compatible with GPU export, i.e. use device=0'
            )
        self.args.half = False
        assert not self.args.dynamic, 'half=True not compatible with dynamic=True, i.e. use only one.'
    self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)
    if self.args.int8 and (engine or xml):
        self.args.dynamic = True
    if self.args.optimize:
        assert not ncnn, "optimize=True not compatible with format='ncnn', i.e. use optimize=False"
        assert self.device.type == 'cpu', "optimize=True not compatible with cuda devices, i.e. use device='cpu'"
    if edgetpu:
        if not LINUX:
            raise SystemError(
                'Edge TPU export only supported on Linux. See https://coral.ai/docs/edgetpu/compiler'
                )
        elif self.args.batch != 1:
            LOGGER.warning(
                'WARNING ⚠️ Edge TPU export requires batch size 1, setting batch=1.'
                )
            self.args.batch = 1
    if isinstance(model, WorldModel):
        LOGGER.warning(
            """WARNING ⚠️ YOLOWorld (original version) export is not supported to any format.
WARNING ⚠️ YOLOWorldv2 models (i.e. 'yolov8s-worldv2.pt') only support export to (torchscript, onnx, openvino, engine, coreml) formats. See https://docs.ultralytics.com/models/yolo-world for details."""
            )
    if self.args.int8 and not self.args.data:
        self.args.data = DEFAULT_CFG.data or TASK2DATA[getattr(model,
            'task', 'detect')]
        LOGGER.warning(
            f"WARNING ⚠️ INT8 export requires a missing 'data' arg for calibration. Using default 'data={self.args.data}'."
            )
    im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)
    file = Path(getattr(model, 'pt_path', None) or getattr(model,
        'yaml_file', None) or model.yaml.get('yaml_file', ''))
    if file.suffix in {'.yaml', '.yml'}:
        file = Path(file.name)
    model = deepcopy(model).to(self.device)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    model = model.fuse()
    for m in model.modules():
        if isinstance(m, (Detect, RTDETRDecoder)):
            m.dynamic = self.args.dynamic
            m.export = True
            m.format = self.args.format
        elif isinstance(m, C2f) and not is_tf_format:
            m.forward = m.forward_split
    y = None
    for _ in range(2):
        y = model(im)
    if self.args.half and onnx and self.device.type != 'cpu':
        im, model = im.half(), model.half()
    warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    self.im = im
    self.model = model
    self.file = file
    self.output_shape = tuple(y.shape) if isinstance(y, torch.Tensor
        ) else tuple(tuple(x.shape if isinstance(x, torch.Tensor) else []) for
        x in y)
    self.pretty_name = Path(self.model.yaml.get('yaml_file', self.file)
        ).stem.replace('yolo', 'YOLO')
    data = model.args['data'] if hasattr(model, 'args') and isinstance(model
        .args, dict) else ''
    description = (
        f"Ultralytics {self.pretty_name} model {f'trained on {data}' if data else ''}"
        )
    self.metadata = {'description': description, 'author': 'Ultralytics',
        'date': datetime.now().isoformat(), 'version': __version__,
        'license': 'AGPL-3.0 License (https://ultralytics.com/license)',
        'docs': 'https://docs.ultralytics.com', 'stride': int(max(model.
        stride)), 'task': model.task, 'batch': self.args.batch, 'imgsz':
        self.imgsz, 'names': model.names}
    if model.task == 'pose':
        self.metadata['kpt_shape'] = model.model[-1].kpt_shape
    LOGGER.info(
        f"""
{colorstr('PyTorch:')} starting from '{file}' with input shape {tuple(im.shape)} BCHW and output shape(s) {self.output_shape} ({file_size(file):.1f} MB)"""
        )
    f = [''] * len(fmts)
    if jit or ncnn:
        f[0], _ = self.export_torchscript()
    if engine:
        f[1], _ = self.export_engine()
    if onnx:
        f[2], _ = self.export_onnx()
    if xml:
        f[3], _ = self.export_openvino()
    if coreml:
        f[4], _ = self.export_coreml()
    if is_tf_format:
        self.args.int8 |= edgetpu
        f[5], keras_model = self.export_saved_model()
        if pb or tfjs:
            f[6], _ = self.export_pb(keras_model=keras_model)
        if tflite:
            f[7], _ = self.export_tflite(keras_model=keras_model, nms=False,
                agnostic_nms=self.args.agnostic_nms)
        if edgetpu:
            f[8], _ = self.export_edgetpu(tflite_model=Path(f[5]) /
                f'{self.file.stem}_full_integer_quant.tflite')
        if tfjs:
            f[9], _ = self.export_tfjs()
    if paddle:
        f[10], _ = self.export_paddle()
    if ncnn:
        f[11], _ = self.export_ncnn()
    f = [str(x) for x in f if x]
    if any(f):
        f = str(Path(f[-1]))
        square = self.imgsz[0] == self.imgsz[1]
        s = ('' if square else
            f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not work. Use export 'imgsz={max(self.imgsz)}' if val is required."
            )
        imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(' ',
            '')
        predict_data = (f'data={data}' if model.task == 'segment' and fmt ==
            'pb' else '')
        q = 'int8' if self.args.int8 else 'half' if self.args.half else ''
        LOGGER.info(
            f"""
Export complete ({time.time() - t:.1f}s)
Results saved to {colorstr('bold', file.parent.resolve())}
Predict:         yolo predict task={model.task} model={f} imgsz={imgsz} {q} {predict_data}
Validate:        yolo val task={model.task} model={f} imgsz={imgsz} data={data} {q} {s}
Visualize:       https://netron.app"""
            )
    self.run_callbacks('on_export_end')
    return f
