def upsample_enable_quant(self, num_bits, calib_method):
    LOGGER.info('Insert fakequant after upsample')
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization.tensor_quant import QuantDescriptor
    conv2d_input_default_desc = QuantDescriptor(num_bits=num_bits,
        calib_method=calib_method)
    self.upsample_feat0_quant = quant_nn.TensorQuantizer(
        conv2d_input_default_desc)
    self.upsample_feat1_quant = quant_nn.TensorQuantizer(
        conv2d_input_default_desc)
    self._QUANT = True
