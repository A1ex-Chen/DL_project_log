def __init__(self, vae: AutoencoderKL, text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer, unet: UNet2DConditionModel, scheduler:
    DDIMScheduler, safety_checker: StableDiffusionSafetyChecker,
    feature_extractor: CLIPFeatureExtractor, image_encoder:
    CLIPVisionModelWithProjection=None, requires_safety_checker: bool=True,
    stages=['clip', 'unet', 'vae', 'vae_encoder'], image_height: int=512,
    image_width: int=512, max_batch_size: int=16, onnx_opset: int=17,
    onnx_dir: str='onnx', engine_dir: str='engine', build_preview_features:
    bool=True, force_engine_rebuild: bool=False, timing_cache: str=
    'timing_cache'):
    super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
        safety_checker=safety_checker, feature_extractor=feature_extractor,
        image_encoder=image_encoder, requires_safety_checker=
        requires_safety_checker)
    self.vae.forward = self.vae.decode
    self.stages = stages
    self.image_height, self.image_width = image_height, image_width
    self.inpaint = True
    self.onnx_opset = onnx_opset
    self.onnx_dir = onnx_dir
    self.engine_dir = engine_dir
    self.force_engine_rebuild = force_engine_rebuild
    self.timing_cache = timing_cache
    self.build_static_batch = False
    self.build_dynamic_shape = False
    self.build_preview_features = build_preview_features
    self.max_batch_size = max_batch_size
    if (self.build_dynamic_shape or self.image_height > 512 or self.
        image_width > 512):
        self.max_batch_size = 4
    self.stream = None
    self.models = {}
    self.engine = {}
