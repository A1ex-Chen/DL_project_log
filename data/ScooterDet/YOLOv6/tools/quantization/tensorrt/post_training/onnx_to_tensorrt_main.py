def main():
    parser = argparse.ArgumentParser(description=
        """Creates a TensorRT engine from the provided ONNX file.
""")
    parser.add_argument('--onnx', required=True, help=
        'The ONNX model file to convert to TensorRT')
    parser.add_argument('-o', '--output', type=str, default='model.engine',
        help='The path at which to write the engine')
    parser.add_argument('-b', '--max-batch-size', type=int, help=
        'The max batch size for the TensorRT engine input')
    parser.add_argument('-v', '--verbosity', action='count', help=
        'Verbosity for logging. (None) for ERROR, (-v) for INFO/WARNING/ERROR, (-vv) for VERBOSE.'
        )
    parser.add_argument('--explicit-batch', action='store_true', help=
        'Set trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH.')
    parser.add_argument('--explicit-precision', action='store_true', help=
        'Set trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION.')
    parser.add_argument('--gpu-fallback', action='store_true', help=
        'Set trt.BuilderFlag.GPU_FALLBACK.')
    parser.add_argument('--refittable', action='store_true', help=
        'Set trt.BuilderFlag.REFIT.')
    parser.add_argument('--debug', action='store_true', help=
        'Set trt.BuilderFlag.DEBUG.')
    parser.add_argument('--strict-types', action='store_true', help=
        'Set trt.BuilderFlag.STRICT_TYPES.')
    parser.add_argument('--fp16', action='store_true', help=
        'Attempt to use FP16 kernels when possible.')
    parser.add_argument('--int8', action='store_true', help=
        'Attempt to use INT8 kernels when possible. This should generally be used in addition to the --fp16 flag.                                                              ONLY SUPPORTS RESNET-LIKE MODELS SUCH AS RESNET50/VGG16/INCEPTION/etc.'
        )
    parser.add_argument('--calibration-cache', help=
        '(INT8 ONLY) The path to read/write from calibration cache.',
        default='calibration.cache')
    parser.add_argument('--calibration-data', help=
        '(INT8 ONLY) The directory containing {*.jpg, *.jpeg, *.png} files to use for calibration. (ex: Imagenet Validation Set)'
        , default=None)
    parser.add_argument('--calibration-batch-size', help=
        '(INT8 ONLY) The batch size to use during calibration.', type=int,
        default=128)
    parser.add_argument('--max-calibration-size', help=
        '(INT8 ONLY) The max number of data to calibrate on from --calibration-data.'
        , type=int, default=2048)
    parser.add_argument('-s', '--simple', action='store_true', help=
        'Use SimpleCalibrator with random data instead of ImagenetCalibrator for INT8 calibration.'
        )
    args, _ = parser.parse_known_args()
    print(args)
    if args.verbosity is None:
        TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    elif args.verbosity == 1:
        TRT_LOGGER.min_severity = trt.Logger.Severity.INFO
    else:
        TRT_LOGGER.min_severity = trt.Logger.Severity.VERBOSE
    logger.info('TRT_LOGGER Verbosity: {:}'.format(TRT_LOGGER.min_severity))
    network_flags = 0
    if args.explicit_batch:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.
            EXPLICIT_BATCH)
    if args.explicit_precision:
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.
            EXPLICIT_PRECISION)
    builder_flag_map = {'gpu_fallback': trt.BuilderFlag.GPU_FALLBACK,
        'refittable': trt.BuilderFlag.REFIT, 'debug': trt.BuilderFlag.DEBUG,
        'strict_types': trt.BuilderFlag.STRICT_TYPES, 'fp16': trt.
        BuilderFlag.FP16, 'int8': trt.BuilderFlag.INT8}
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        network_flags) as network, builder.create_builder_config(
        ) as config, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = 2 ** 30
        for flag in builder_flag_map:
            if getattr(args, flag):
                logger.info('Setting {}'.format(builder_flag_map[flag]))
                config.set_flag(builder_flag_map[flag])
        with open(args.onnx, 'rb') as f:
            if not parser.parse(f.read()):
                print('ERROR: Failed to parse the ONNX file: {}'.format(
                    args.onnx))
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                sys.exit(1)
        check_network(network)
        if args.explicit_batch:
            batch_sizes = [1, 8, 16, 32, 64]
            inputs = [network.get_input(i) for i in range(network.num_inputs)]
            opt_profiles = create_optimization_profiles(builder, inputs,
                batch_sizes)
            add_profiles(config, inputs, opt_profiles)
        else:
            builder.max_batch_size = args.max_batch_size
            opt_profiles = []
        if args.fp16 and not builder.platform_has_fast_fp16:
            logger.warning('FP16 not supported on this platform.')
        if args.int8 and not builder.platform_has_fast_int8:
            logger.warning('INT8 not supported on this platform.')
        if args.int8:
            from Calibrator import ImageCalibrator, get_int8_calibrator
            config.int8_calibrator = get_int8_calibrator(args.
                calibration_cache, args.calibration_data, args.
                max_calibration_size, args.calibration_batch_size)
        logger.info('Building Engine...')
        with builder.build_engine(network, config) as engine, open(args.
            output, 'wb') as f:
            logger.info('Serializing engine to file: {:}'.format(args.output))
            f.write(engine.serialize())
