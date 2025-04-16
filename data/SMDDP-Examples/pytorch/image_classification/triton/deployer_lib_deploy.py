def deploy(self, dataloader, model):
    """ deploy the model and test for correctness with dataloader """
    if self.args.ts_script or self.args.ts_trace:
        self.lib.set_platform('pytorch_libtorch')
        print('deploying model ' + self.args.triton_model_name +
            ' in format ' + self.lib.platform)
        self.to_triton_torchscript(dataloader, model)
    elif self.args.onnx:
        self.lib.set_platform('onnxruntime_onnx')
        print('deploying model ' + self.args.triton_model_name +
            ' in format ' + self.lib.platform)
        self.to_triton_onnx(dataloader, model)
    elif self.args.trt:
        self.lib.set_platform('tensorrt_plan')
        print('deploying model ' + self.args.triton_model_name +
            ' in format ' + self.lib.platform)
        self.to_triton_trt(dataloader, model)
    else:
        assert False, 'error'
    print('done')
