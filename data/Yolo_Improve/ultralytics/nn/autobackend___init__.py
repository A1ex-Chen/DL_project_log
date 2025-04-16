@torch.no_grad()
def __init__(self, weights='yolov8n.pt', device=torch.device('cpu'), dnn=
    False, data=None, fp16=False, batch=1, fuse=True, verbose=True):
    """
        Initialize the AutoBackend for inference.

        Args:
            weights (str): Path to the model weights file. Defaults to 'yolov8n.pt'.
            device (torch.device): Device to run the model on. Defaults to CPU.
            dnn (bool): Use OpenCV DNN module for ONNX inference. Defaults to False.
            data (str | Path | optional): Path to the additional data.yaml file containing class names. Optional.
            fp16 (bool): Enable half-precision inference. Supported only on specific backends. Defaults to False.
            batch (int): Batch-size to assume for inference.
            fuse (bool): Fuse Conv2D + BatchNorm layers for optimization. Defaults to True.
            verbose (bool): Enable verbose logging. Defaults to True.
        """
    super().__init__()
    w = str(weights[0] if isinstance(weights, list) else weights)
    nn_module = isinstance(weights, torch.nn.Module)
    (pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu,
        tfjs, paddle, ncnn, triton) = self._model_type(w)
    fp16 &= pt or jit or onnx or xml or engine or nn_module or triton
    nhwc = coreml or saved_model or pb or tflite or edgetpu
    stride = 32
    model, metadata = None, None
    cuda = torch.cuda.is_available() and device.type != 'cpu'
    if cuda and not any([nn_module, pt, jit, engine, onnx]):
        device = torch.device('cpu')
        cuda = False
    if not (pt or triton or nn_module):
        w = attempt_download_asset(w)
    if nn_module:
        model = weights.to(device)
        if fuse:
            model = model.fuse(verbose=verbose)
        if hasattr(model, 'kpt_shape'):
            kpt_shape = model.kpt_shape
        stride = max(int(model.stride.max()), 32)
        names = model.module.names if hasattr(model, 'module') else model.names
        model.half() if fp16 else model.float()
        self.model = model
        pt = True
    elif pt:
        from ultralytics.nn.tasks import attempt_load_weights
        model = attempt_load_weights(weights if isinstance(weights, list) else
            w, device=device, inplace=True, fuse=fuse)
        if hasattr(model, 'kpt_shape'):
            kpt_shape = model.kpt_shape
        stride = max(int(model.stride.max()), 32)
        names = model.module.names if hasattr(model, 'module') else model.names
        model.half() if fp16 else model.float()
        self.model = model
    elif jit:
        LOGGER.info(f'Loading {w} for TorchScript inference...')
        extra_files = {'config.txt': ''}
        model = torch.jit.load(w, _extra_files=extra_files, map_location=device
            )
        model.half() if fp16 else model.float()
        if extra_files['config.txt']:
            metadata = json.loads(extra_files['config.txt'], object_hook=lambda
                x: dict(x.items()))
    elif dnn:
        LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
        check_requirements('opencv-python>=4.5.4')
        net = cv2.dnn.readNetFromONNX(w)
    elif onnx:
        LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
        check_requirements(('onnx', 'onnxruntime-gpu' if cuda else
            'onnxruntime'))
        if IS_RASPBERRYPI or IS_JETSON:
            check_requirements('numpy==1.23.5')
        import onnxruntime
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'
            ] if cuda else ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(w, providers=providers)
        output_names = [x.name for x in session.get_outputs()]
        metadata = session.get_modelmeta().custom_metadata_map
    elif xml:
        LOGGER.info(f'Loading {w} for OpenVINO inference...')
        check_requirements('openvino>=2024.0.0')
        import openvino as ov
        core = ov.Core()
        w = Path(w)
        if not w.is_file():
            w = next(w.glob('*.xml'))
        ov_model = core.read_model(model=str(w), weights=w.with_suffix('.bin'))
        if ov_model.get_parameters()[0].get_layout().empty:
            ov_model.get_parameters()[0].set_layout(ov.Layout('NCHW'))
        inference_mode = 'CUMULATIVE_THROUGHPUT' if batch > 1 else 'LATENCY'
        LOGGER.info(
            f'Using OpenVINO {inference_mode} mode for batch={batch} inference...'
            )
        ov_compiled_model = core.compile_model(ov_model, device_name='AUTO',
            config={'PERFORMANCE_HINT': inference_mode})
        input_name = ov_compiled_model.input().get_any_name()
        metadata = w.parent / 'metadata.yaml'
    elif engine:
        LOGGER.info(f'Loading {w} for TensorRT inference...')
        try:
            import tensorrt as trt
        except ImportError:
            if LINUX:
                check_requirements('tensorrt>7.0.0,<=10.1.0')
            import tensorrt as trt
        check_version(trt.__version__, '>=7.0.0', hard=True)
        check_version(trt.__version__, '<=10.1.0', msg=
            'https://github.com/ultralytics/ultralytics/pull/14239')
        if device.type == 'cpu':
            device = torch.device('cuda:0')
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data',
            'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
            try:
                meta_len = int.from_bytes(f.read(4), byteorder='little')
                metadata = json.loads(f.read(meta_len).decode('utf-8'))
            except UnicodeDecodeError:
                f.seek(0)
            model = runtime.deserialize_cuda_engine(f.read())
        try:
            context = model.create_execution_context()
        except Exception as e:
            LOGGER.error(
                f"""ERROR: TensorRT model exported with a different version than {trt.__version__}
"""
                )
            raise e
        bindings = OrderedDict()
        output_names = []
        fp16 = False
        dynamic = False
        is_trt10 = not hasattr(model, 'num_bindings')
        num = range(model.num_io_tensors) if is_trt10 else range(model.
            num_bindings)
        for i in num:
            if is_trt10:
                name = model.get_tensor_name(i)
                dtype = trt.nptype(model.get_tensor_dtype(name))
                is_input = model.get_tensor_mode(name
                    ) == trt.TensorIOMode.INPUT
                if is_input:
                    if -1 in tuple(model.get_tensor_shape(name)):
                        dynamic = True
                        context.set_input_shape(name, tuple(model.
                            get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_tensor_shape(name))
            else:
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                is_input = model.binding_is_input(i)
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.
                            get_profile_shape(0, i)[1]))
                    if dtype == np.float16:
                        fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr())
                )
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings['images'].shape[0]
    elif coreml:
        LOGGER.info(f'Loading {w} for CoreML inference...')
        import coremltools as ct
        model = ct.models.MLModel(w)
        metadata = dict(model.user_defined_metadata)
    elif saved_model:
        LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
        import tensorflow as tf
        keras = False
        model = tf.keras.models.load_model(w
            ) if keras else tf.saved_model.load(w)
        metadata = Path(w) / 'metadata.yaml'
    elif pb:
        LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
        import tensorflow as tf
        from ultralytics.engine.exporter import gd_outputs

        def wrap_frozen_graph(gd, inputs, outputs):
            """Wrap frozen graphs for deployment."""
            x = tf.compat.v1.wrap_function(lambda : tf.compat.v1.
                import_graph_def(gd, name=''), [])
            ge = x.graph.as_graph_element
            return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.
                map_structure(ge, outputs))
        gd = tf.Graph().as_graph_def()
        with open(w, 'rb') as f:
            gd.ParseFromString(f.read())
        frozen_func = wrap_frozen_graph(gd, inputs='x:0', outputs=
            gd_outputs(gd))
        with contextlib.suppress(StopIteration):
            metadata = next(Path(w).resolve().parent.rglob(
                f'{Path(w).stem}_saved_model*/metadata.yaml'))
    elif tflite or edgetpu:
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
        except ImportError:
            import tensorflow as tf
            Interpreter, load_delegate = (tf.lite.Interpreter, tf.lite.
                experimental.load_delegate)
        if edgetpu:
            LOGGER.info(
                f'Loading {w} for TensorFlow Lite Edge TPU inference...')
            delegate = {'Linux': 'libedgetpu.so.1', 'Darwin':
                'libedgetpu.1.dylib', 'Windows': 'edgetpu.dll'}[platform.
                system()]
            interpreter = Interpreter(model_path=w, experimental_delegates=
                [load_delegate(delegate)])
        else:
            LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
            interpreter = Interpreter(model_path=w)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        with contextlib.suppress(zipfile.BadZipFile):
            with zipfile.ZipFile(w, 'r') as model:
                meta_file = model.namelist()[0]
                metadata = ast.literal_eval(model.read(meta_file).decode(
                    'utf-8'))
    elif tfjs:
        raise NotImplementedError(
            'YOLOv8 TF.js inference is not currently supported.')
    elif paddle:
        LOGGER.info(f'Loading {w} for PaddlePaddle inference...')
        check_requirements('paddlepaddle-gpu' if cuda else 'paddlepaddle')
        import paddle.inference as pdi
        w = Path(w)
        if not w.is_file():
            w = next(w.rglob('*.pdmodel'))
        config = pdi.Config(str(w), str(w.with_suffix('.pdiparams')))
        if cuda:
            config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
        predictor = pdi.create_predictor(config)
        input_handle = predictor.get_input_handle(predictor.get_input_names
            ()[0])
        output_names = predictor.get_output_names()
        metadata = w.parents[1] / 'metadata.yaml'
    elif ncnn:
        LOGGER.info(f'Loading {w} for NCNN inference...')
        check_requirements('git+https://github.com/Tencent/ncnn.git' if
            ARM64 else 'ncnn')
        import ncnn as pyncnn
        net = pyncnn.Net()
        net.opt.use_vulkan_compute = cuda
        w = Path(w)
        if not w.is_file():
            w = next(w.glob('*.param'))
        net.load_param(str(w))
        net.load_model(str(w.with_suffix('.bin')))
        metadata = w.parent / 'metadata.yaml'
    elif triton:
        check_requirements('tritonclient[all]')
        from ultralytics.utils.triton import TritonRemoteModel
        model = TritonRemoteModel(w)
    else:
        from ultralytics.engine.exporter import export_formats
        raise TypeError(
            f"""model='{w}' is not a supported model format. See https://docs.ultralytics.com/modes/predict for help.

{export_formats()}"""
            )
    if isinstance(metadata, (str, Path)) and Path(metadata).exists():
        metadata = yaml_load(metadata)
    if metadata and isinstance(metadata, dict):
        for k, v in metadata.items():
            if k in {'stride', 'batch'}:
                metadata[k] = int(v)
            elif k in {'imgsz', 'names', 'kpt_shape'} and isinstance(v, str):
                metadata[k] = eval(v)
        stride = metadata['stride']
        task = metadata['task']
        batch = metadata['batch']
        imgsz = metadata['imgsz']
        names = metadata['names']
        kpt_shape = metadata.get('kpt_shape')
    elif not (pt or triton or nn_module):
        LOGGER.warning(f"WARNING ⚠️ Metadata not found for 'model={weights}'")
    if 'names' not in locals():
        names = default_class_names(data)
    names = check_class_names(names)
    if pt:
        for p in model.parameters():
            p.requires_grad = False
    self.__dict__.update(locals())
