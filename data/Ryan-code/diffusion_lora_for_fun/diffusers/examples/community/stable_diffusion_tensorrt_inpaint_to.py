def to(self, torch_device: Optional[Union[str, torch.device]]=None,
    silence_dtype_warnings: bool=False):
    super().to(torch_device, silence_dtype_warnings=silence_dtype_warnings)
    self.onnx_dir = os.path.join(self.cached_folder, self.onnx_dir)
    self.engine_dir = os.path.join(self.cached_folder, self.engine_dir)
    self.timing_cache = os.path.join(self.cached_folder, self.timing_cache)
    self.torch_device = self._execution_device
    logger.warning(f'Running inference on device: {self.torch_device}')
    self.__loadModels()
    self.engine = build_engines(self.models, self.engine_dir, self.onnx_dir,
        self.onnx_opset, opt_image_height=self.image_height,
        opt_image_width=self.image_width, force_engine_rebuild=self.
        force_engine_rebuild, static_batch=self.build_static_batch,
        static_shape=not self.build_dynamic_shape, enable_preview=self.
        build_preview_features, timing_cache=self.timing_cache)
    return self
