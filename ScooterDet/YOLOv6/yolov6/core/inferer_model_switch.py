def model_switch(self, model, img_size):
    """ Model switch to deploy status """
    from yolov6.layers.common import RepVGGBlock
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer,
            'recompute_scale_factor'):
            layer.recompute_scale_factor = None
    LOGGER.info('Switch model to deploy modality.')
