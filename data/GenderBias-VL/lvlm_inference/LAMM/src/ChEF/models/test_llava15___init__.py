def __init__(self, model_path, model_base=None, vis_processor_path=None,
    device='cuda', **kwargs):
    model_name = get_model_name_from_path(model_path)
    print(f'Load model on device map: {device}')
    self.tokenizer, self.model, self.image_processor, self.context_len = (
        load_pretrained_model(model_path, model_base, model_name,
        vis_processor_path=vis_processor_path, device=device))
    self.conv = get_conv(model_name)
    self.stop_str = [self.conv.sep if self.conv.sep_style != SeparatorStyle
        .TWO else self.conv.sep2]
    self.image_process_mode = 'Resize'
    self.model.eval()
    self.dtype = self.model.dtype
    self.device = self.model.device
