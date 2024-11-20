def main():
    """Create a TensorRT engine for ONNX-based YOLO."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help=
        'enable verbose output (for debugging)')
    parser.add_argument('-m', '--model', type=str, required=True, help=
        'onnx model path')
    parser.add_argument('-d', '--dtype', type=str, required=True, help=
        'one type of int8, fp16, fp32')
    parser.add_argument('--qat', action='store_true', help=
        'whether the onnx model is qat; if it is, the int8 calibrator is not needed'
        )
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 
        640], help='image size of model input, the order is: height width')
    parser.add_argument('--batch-size', type=int, default=128, help=
        'batch size for training: default 64')
    parser.add_argument('--num-calib-batch', default=6, type=int, help=
        'Number of batches for calibration')
    parser.add_argument('--calib-img-dir', default=
        '../coco/images/train2017', type=str, help=
        'Number of batches for calibration')
    parser.add_argument('--calib-cache', default=
        './yolov6s_calibration.cache', type=str, help=
        'Path of calibration cache')
    args = parser.parse_args()
    if args.dtype == 'int8' and not args.qat:
        from calibrator import DataLoader, Calibrator
        if len(args.img_size) == 1:
            args.img_size = [args.img_size[0], args.img_size[0]]
        calib_loader = DataLoader(args.batch_size, args.num_calib_batch,
            args.calib_img_dir, args.img_size[1], args.img_size[0])
        engine = build_engine_from_onnx(args.model, args.dtype, args.
            verbose, int8_calib=True, calib_loader=calib_loader,
            calib_cache=args.calib_cache)
    else:
        engine = build_engine_from_onnx(args.model, args.dtype, args.verbose)
    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')
    engine_path = args.model.replace('.onnx', '.trt')
    if args.dtype == 'int8' and not args.qat:
        engine_path = args.model.replace('.onnx', '-int8-{}-{}-minmax.trt'.
            format(args.batch_size, args.num_calib_batch))
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_path)
