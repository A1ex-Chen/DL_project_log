def set_platform(self, platform):
    """ sets the platform
            :: platform :: "pytorch_libtorch" or "onnxruntime_onnx" or "tensorrt_plan"
        """
    self.platform = platform
