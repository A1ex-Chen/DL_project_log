def serialize(ov_model, file):
    """Set RT info, serialize and save metadata YAML."""
    ov_model.set_rt_info('YOLOv8', ['model_info', 'model_type'])
    ov_model.set_rt_info(True, ['model_info', 'reverse_input_channels'])
    ov_model.set_rt_info(114, ['model_info', 'pad_value'])
    ov_model.set_rt_info([255.0], ['model_info', 'scale_values'])
    ov_model.set_rt_info(self.args.iou, ['model_info', 'iou_threshold'])
    ov_model.set_rt_info([v.replace(' ', '_') for v in self.model.names.
        values()], ['model_info', 'labels'])
    if self.model.task != 'classify':
        ov_model.set_rt_info('fit_to_window_letterbox', ['model_info',
            'resize_type'])
    ov.runtime.save_model(ov_model, file, compress_to_fp16=self.args.half)
    yaml_save(Path(file).parent / 'metadata.yaml', self.metadata)
