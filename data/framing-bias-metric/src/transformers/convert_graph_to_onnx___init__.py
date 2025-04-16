def __init__(self):
    super().__init__('ONNX Converter')
    self.add_argument('--pipeline', type=str, choices=SUPPORTED_PIPELINES,
        default='feature-extraction')
    self.add_argument('--model', type=str, required=True, help=
        "Model's id or path (ex: bert-base-cased)")
    self.add_argument('--tokenizer', type=str, help=
        "Tokenizer's id or path (ex: bert-base-cased)")
    self.add_argument('--framework', type=str, choices=['pt', 'tf'], help=
        'Framework for loading the model')
    self.add_argument('--opset', type=int, default=11, help='ONNX opset to use'
        )
    self.add_argument('--check-loading', action='store_true', help=
        'Check ONNX is able to load the model')
    self.add_argument('--use-external-format', action='store_true', help=
        'Allow exporting model >= than 2Gb')
    self.add_argument('--quantize', action='store_true', help=
        'Quantize the neural network to be run with int8')
    self.add_argument('output')
