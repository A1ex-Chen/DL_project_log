def create_deployer(argv):
    """ takes a list of arguments, returns a deployer object and the list of unused arguments """
    parser = argparse.ArgumentParser()
    method = parser.add_mutually_exclusive_group(required=True)
    method.add_argument('--ts-script', action='store_true', help=
        'convert to torchscript using torch.jit.script')
    method.add_argument('--ts-trace', action='store_true', help=
        'convert to torchscript using torch.jit.trace')
    method.add_argument('--onnx', action='store_true', help=
        'convert to onnx using torch.onnx.export')
    method.add_argument('--trt', action='store_true', help=
        'convert to trt using tensorrt')
    arguments = parser.add_argument_group('triton related flags')
    arguments.add_argument('--triton-no-cuda', action='store_true', help=
        'Use the CPU for tracing.')
    arguments.add_argument('--triton-model-name', type=str, default='model',
        help='exports to appropriate directory structure for TRITON')
    arguments.add_argument('--triton-model-version', type=int, default=1,
        help='exports to appropriate directory structure for TRITON')
    arguments.add_argument('--triton-max-batch-size', type=int, default=8,
        help=
        "Specifies the 'max_batch_size' in the TRITON model config.                                  See the TRITON documentation for more info."
        )
    arguments.add_argument('--triton-dyn-batching-delay', type=float,
        default=0, help=
        "Determines the dynamic_batching queue delay in milliseconds(ms) for                                  the TRITON model config. Use '0' or '-1' to specify static batching.                                  See the TRITON documentation for more info."
        )
    arguments.add_argument('--triton-engine-count', type=int, default=1,
        help=
        "Specifies the 'instance_group' count value in the TRITON model config.                                  See the TRITON documentation for more info."
        )
    arguments.add_argument('--save-dir', type=str, default=
        './triton_models', help='Saved model directory')
    arguments = parser.add_argument_group('optimization flags')
    arguments.add_argument('--max_workspace_size', type=int, default=512 * 
        1024 * 1024, help='set the size of the workspace for trt export')
    arguments.add_argument('--trt-fp16', action='store_true', help=
        'trt flag ---- export model in mixed precision mode')
    arguments.add_argument('--capture-cuda-graph', type=int, default=1,
        help=
        'capture cuda graph for obtaining speedup. possible values: 0, 1. default: 1. '
        )
    arguments.add_argument('model_arguments', nargs=argparse.REMAINDER,
        help=
        'arguments that will be ignored by deployer lib and will be forwarded to your deployer script'
        )
    args = parser.parse_args(argv)
    deployer = Deployer(args)
    return deployer, args.model_arguments[1:]
