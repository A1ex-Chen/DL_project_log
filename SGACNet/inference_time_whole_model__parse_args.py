def _parse_args():
    parser = ArgumentParserRGBDSegmentation(description=
        'Efficient RGBD Indoor Sematic Segmentation', formatter_class=
        argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--model', type=str, default='own', choices=['own',
        'onnx'], help='The model for which the inference time will bemeasured.'
        )
    parser.add_argument('--model_onnx_filepath', type=str, default=None,
        help="Path to ONNX model file when --model is 'onnx'")
    parser.add_argument('--n_runs', type=int, default=100, help=
        'For how many runs the inference time will be measured')
    parser.add_argument('--n_runs_warmup', type=int, default=10, help=
        'How many forward paths trough the model beforethe inference time measurements starts. This is necessary as the first runs are slower.'
        )
    parser.add_argument('--no_time_pytorch', dest='time_pytorch', action=
        'store_false', default=True, help=
        'Set this if you do not want to measure thepytorch times.')
    parser.add_argument('--no_time_tensorrt', dest='time_tensorrt', action=
        'store_false', default=True, help=
        'Set this if you do not want to measure the tensorrt times.')
    parser.add_argument('--no_time_onnxruntime', dest='time_onnxruntime',
        action='store_false', default=True, help=
        'Set this if you do not want to measure the onnxruntime times.')
    parser.add_argument('--plot_timing', default=False, action='store_true',
        help='Wether to plot the inference time for eachforward pass')
    parser.add_argument('--plot_outputs', default=False, action=
        'store_true', help=
        'Wether to plot the colored segmentation outputof the model')
    parser.add_argument('--export_outputs', default=False, action=
        'store_true', help=
        'Whether to export the colored segmentation outputof the model to png')
    parser.add_argument('--trt_workspace', type=int, default=2 << 30, help=
        'default is 2GB')
    parser.add_argument('--trt_floatx', type=int, default=32, choices=[16, 
        32], help='Whether to measure tensorrt timings with float16or float32.'
        )
    parser.add_argument('--trt_batchsize', type=int, default=1)
    parser.add_argument('--trt_onnx_opset_version', type=int, default=10,
        help=
        'different versions lead to different results butnot all versions are supported for the followingtensorrt conversion.'
        )
    parser.add_argument('--trt_dont_force_rebuild', dest=
        'trt_force_rebuild', default=True, action='store_false', help=
        'Possibly already existing trt engine file will be reused when providing this argument.'
        )
    parser.add_argument('--onnxruntime_onnx_opset_version', type=int,
        default=11, help=
        'opset 10 leads to different results compared toPyTorch')
    parser.add_argument('--onnxruntime_trt_max_partition_iterations', type=
        str, default='15', help=
        'maximum number of iterations allowed in model partitioning for TensorRT'
        )
    args = parser.parse_args()
    args.pretrained_on_imagenet = False
    return args
