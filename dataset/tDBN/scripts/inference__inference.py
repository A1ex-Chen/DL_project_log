def _inference(self, example):
    train_cfg = self.config.train_config
    input_cfg = self.config.eval_input_reader
    model_cfg = self.config.model.tDBN
    example_torch = example_convert_to_torch(example)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32
    result_annos = predict_kitti_to_anno(self.net, example_torch, list(
        input_cfg.class_names), model_cfg.post_center_limit_range,
        model_cfg.lidar_input)
    return result_annos
