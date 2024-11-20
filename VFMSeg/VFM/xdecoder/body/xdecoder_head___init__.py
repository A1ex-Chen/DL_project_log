@configurable
def __init__(self, input_shape: Dict[str, ShapeSpec], *, num_classes: int,
    pixel_decoder: nn.Module, loss_weight: float=1.0, ignore_value: int=-1,
    transformer_predictor: nn.Module, transformer_in_feature: str):
    """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
    super().__init__()
    input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
    self.in_features = [k for k, v in input_shape]
    feature_strides = [v.stride for k, v in input_shape]
    feature_channels = [v.channels for k, v in input_shape]
    self.ignore_value = ignore_value
    self.common_stride = 4
    self.loss_weight = loss_weight
    self.pixel_decoder = pixel_decoder
    self.predictor = transformer_predictor
    self.transformer_in_feature = transformer_in_feature
    self.num_classes = num_classes
