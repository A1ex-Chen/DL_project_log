@try_export
def export_openvino(file, metadata, half, int8, data, prefix=colorstr(
    'OpenVINO:')):
    check_requirements('openvino-dev>=2023.0')
    import openvino.runtime as ov
    from openvino.tools import mo
    LOGGER.info(f'\n{prefix} starting export with openvino {ov.__version__}...'
        )
    f = str(file).replace(file.suffix, f'_openvino_model{os.sep}')
    f_onnx = file.with_suffix('.onnx')
    f_ov = str(Path(f) / file.with_suffix('.xml').name)
    if int8:
        check_requirements('nncf>=2.4.0')
        import nncf
        import numpy as np
        from openvino.runtime import Core
        from utils.dataloaders import create_dataloader
        core = Core()
        onnx_model = core.read_model(f_onnx)

        def prepare_input_tensor(image: np.ndarray):
            input_tensor = image.astype(np.float32)
            input_tensor /= 255.0
            if input_tensor.ndim == 3:
                input_tensor = np.expand_dims(input_tensor, 0)
            return input_tensor

        def gen_dataloader(yaml_path, task='train', imgsz=640, workers=4):
            data_yaml = check_yaml(yaml_path)
            data = check_dataset(data_yaml)
            dataloader = create_dataloader(data[task], imgsz=imgsz,
                batch_size=1, stride=32, pad=0.5, single_cls=False, rect=
                False, workers=workers)[0]
            return dataloader

        def transform_fn(data_item):
            """
            Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
            Parameters:
               data_item: Tuple with data item produced by DataLoader during iteration
            Returns:
                input_tensor: Input data for quantization
            """
            img = data_item[0].numpy()
            input_tensor = prepare_input_tensor(img)
            return input_tensor
        ds = gen_dataloader(data)
        quantization_dataset = nncf.Dataset(ds, transform_fn)
        ov_model = nncf.quantize(onnx_model, quantization_dataset, preset=
            nncf.QuantizationPreset.MIXED)
    else:
        ov_model = mo.convert_model(f_onnx, model_name=file.stem, framework
            ='onnx', compress_to_fp16=half)
    ov.serialize(ov_model, f_ov)
    yaml_save(Path(f) / file.with_suffix('.yaml').name, metadata)
    return f, None
