def init_model(self, model, weights, task):
    if task != 'train':
        if not os.path.exists(weights):
            download_ckpt(weights)
        model = load_checkpoint(weights, map_location=self.device)
        self.stride = int(model.stride.max())
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer,
                'recompute_scale_factor'):
                layer.recompute_scale_factor = None
        LOGGER.info('Switch model to deploy modality.')
        LOGGER.info('Model Summary: {}'.format(get_model_info(model, self.
            img_size)))
    if self.device.type != 'cpu':
        model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.
            device).type_as(next(model.parameters())))
    model.half() if self.half else model.float()
    return model
