@try_export
def export_openvino(self, prefix=colorstr('OpenVINO:')):
    """YOLOv8 OpenVINO export."""
    check_requirements(f"openvino{'<=2024.0.0' if ARM64 else '>=2024.0.0'}")
    import openvino as ov
    LOGGER.info(f'\n{prefix} starting export with openvino {ov.__version__}...'
        )
    assert TORCH_1_13, f'OpenVINO export requires torch>=1.13.0 but torch=={torch.__version__} is installed'
    ov_model = ov.convert_model(self.model, input=None if self.args.dynamic
         else [self.im.shape], example_input=self.im)

    def serialize(ov_model, file):
        """Set RT info, serialize and save metadata YAML."""
        ov_model.set_rt_info('YOLOv8', ['model_info', 'model_type'])
        ov_model.set_rt_info(True, ['model_info', 'reverse_input_channels'])
        ov_model.set_rt_info(114, ['model_info', 'pad_value'])
        ov_model.set_rt_info([255.0], ['model_info', 'scale_values'])
        ov_model.set_rt_info(self.args.iou, ['model_info', 'iou_threshold'])
        ov_model.set_rt_info([v.replace(' ', '_') for v in self.model.names
            .values()], ['model_info', 'labels'])
        if self.model.task != 'classify':
            ov_model.set_rt_info('fit_to_window_letterbox', ['model_info',
                'resize_type'])
        ov.runtime.save_model(ov_model, file, compress_to_fp16=self.args.half)
        yaml_save(Path(file).parent / 'metadata.yaml', self.metadata)
    if self.args.int8:
        fq = str(self.file).replace(self.file.suffix,
            f'_int8_openvino_model{os.sep}')
        fq_ov = str(Path(fq) / self.file.with_suffix('.xml').name)
        check_requirements('nncf>=2.8.0')
        import nncf

        def transform_fn(data_item) ->np.ndarray:
            """Quantization transform function."""
            data_item: torch.Tensor = data_item['img'] if isinstance(data_item,
                dict) else data_item
            assert data_item.dtype == torch.uint8, 'Input image must be uint8 for the quantization preprocessing'
            im = data_item.numpy().astype(np.float32) / 255.0
            return np.expand_dims(im, 0) if im.ndim == 3 else im
        ignored_scope = None
        if isinstance(self.model.model[-1], Detect):
            head_module_name = '.'.join(list(self.model.named_modules())[-1
                ][0].split('.')[:2])
            ignored_scope = nncf.IgnoredScope(patterns=[
                f'.*{head_module_name}/.*/Add',
                f'.*{head_module_name}/.*/Sub*',
                f'.*{head_module_name}/.*/Mul*',
                f'.*{head_module_name}/.*/Div*',
                f'.*{head_module_name}\\.dfl.*'], types=['Sigmoid'])
        quantized_ov_model = nncf.quantize(model=ov_model,
            calibration_dataset=nncf.Dataset(self.
            get_int8_calibration_dataloader(prefix), transform_fn), preset=
            nncf.QuantizationPreset.MIXED, ignored_scope=ignored_scope)
        serialize(quantized_ov_model, fq_ov)
        return fq, None
    f = str(self.file).replace(self.file.suffix, f'_openvino_model{os.sep}')
    f_ov = str(Path(f) / self.file.with_suffix('.xml').name)
    serialize(ov_model, f_ov)
    return f, None
