def export_onnx(self):
    """
        Export the model to ONNX format.
        Note that the exported model contains custom ops only available in caffe2, therefore it
        cannot be directly executed by other runtime (such as onnxruntime or TensorRT).
        Post-processing or transformation passes may be applied on the model to accommodate
        different runtimes, but we currently do not provide support for them.

        Returns:
            onnx.ModelProto: an onnx model.
        """
    from .caffe2_export import export_onnx_model as export_onnx_model_impl
    return export_onnx_model_impl(self.traceable_model, (self.
        traceable_inputs,))
