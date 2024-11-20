def main():
    args = parse_args()
    check_args(args)
    assert args.model.endswith('.trt'), 'Only support trt engine test'
    processor = Processor(model=args.model)
    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    generate_results(processor, args.imgs_dir, args.visual_dir, jpgs, args.
        conf_thres, args.iou_thres, batch_size=args.batch_size, img_size=
        args.img_size, shrink_size=args.shrink_size)
