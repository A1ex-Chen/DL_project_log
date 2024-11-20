def __init__(self, device, cfg_path=
    'ChEF/models/minigpt4_v2/minigptv2_eval.yaml', **kwargs):
    cfg = Config(cfg_path, None)
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    vis_processor_cfg = cfg.preprocess_cfg.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name
        ).from_config(vis_processor_cfg)
    model.eval()
    self.model, self.vis_processor = model, vis_processor
    print(self.vis_processor.transform)
    self.model.llama_model = self.model.llama_model.float().to(device)
    self.dtype = torch.float16
    self.device = device
    self.model = self.model.to(self.device, dtype=self.dtype)
    self.chat = Chat(model, vis_processor, device=self.device)
