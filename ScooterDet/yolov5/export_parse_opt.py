def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT /
        'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT /
        'yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=
        int, default=[640, 640], help='image (h, w)')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='cpu', help=
        'cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', action='store_true', help=
        'FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help=
        'set YOLOv5 Detect() inplace=True')
    parser.add_argument('--keras', action='store_true', help='TF: use Keras')
    parser.add_argument('--optimize', action='store_true', help=
        'TorchScript: optimize for mobile')
    parser.add_argument('--int8', action='store_true', help=
        'CoreML/TF/OpenVINO INT8 quantization')
    parser.add_argument('--per-tensor', action='store_true', help=
        'TF per-tensor quantization')
    parser.add_argument('--dynamic', action='store_true', help=
        'ONNX/TF/TensorRT: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help=
        'ONNX: simplify model')
    parser.add_argument('--opset', type=int, default=17, help=
        'ONNX: opset version')
    parser.add_argument('--verbose', action='store_true', help=
        'TensorRT: verbose log')
    parser.add_argument('--workspace', type=int, default=4, help=
        'TensorRT: workspace size (GB)')
    parser.add_argument('--nms', action='store_true', help=
        'TF: add NMS to model')
    parser.add_argument('--agnostic-nms', action='store_true', help=
        'TF: add agnostic NMS to model')
    parser.add_argument('--topk-per-class', type=int, default=100, help=
        'TF.js NMS: topk per class to keep')
    parser.add_argument('--topk-all', type=int, default=100, help=
        'TF.js NMS: topk for all classes to keep')
    parser.add_argument('--iou-thres', type=float, default=0.45, help=
        'TF.js NMS: IoU threshold')
    parser.add_argument('--conf-thres', type=float, default=0.25, help=
        'TF.js NMS: confidence threshold')
    parser.add_argument('--include', nargs='+', default=['torchscript'],
        help=
        'torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle'
        )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt
