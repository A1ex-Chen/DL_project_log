def get_results():
    parser = ArgumentParserRGBDSegmentation(description=
        'Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str, required=True, help=
        'Path to the checkpoint of the trained model.')
    parser.add_argument('--depth_scale', type=float, default=1.0, help=
        'Additional depth scaling factor to apply.')
    args = parser.parse_args()
    args.pretrained_on_imagenet = False
    dataset, preprocessor = prepare_data(args, with_input_orig=True)
    n_classes = dataset.n_classes_without_void
    model, device = build_model(args, n_classes=n_classes)
    checkpoint = torch.load(args.ckpt_path, map_location=lambda storage,
        loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))
    model.eval()
    model.to(device)
    basepath = '/home/cyxiong/SGACNet/datasets/sunrgbd/'
    f = open('/home/cyxiong/SGACNet/datasets/sunrgbd/train_depth.txt', 'r')
    p = open('/home/cyxiong/SGACNet/datasets/sunrgbd/train_rgb.txt', 'r')
    q = open('/home/cyxiong/SGACNet/datasets/sunrgbd/list.txt', 'r')
    while True:
        line1 = f.readline()
        line1 = line1[:-1]
        line0 = p.readline()
        line0 = line0[:-1]
        line = q.readline()
        line = line[:-1]
        if line1 and line0 and line:
            print(line1)
            print(line0)
            print(line)
            rgb_filepaths = basepath + line0
            depth_filepaths = basepath + line1
            assert args.modality == 'rgbd', 'Only RGBD inference supported so far'
            img_rgb = cv2.imread(rgb_filepaths, cv2.IMREAD_UNCHANGED)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            img_depth = cv2.imread(depth_filepaths, cv2.IMREAD_UNCHANGED)
            img_depth = img_depth.astype('float32') * args.depth_scale
            h, w, _ = img_rgb.shape
            sample = preprocessor({'image': img_rgb, 'depth': img_depth})
            image = sample['image'][None].to(device)
            depth = sample['depth'][None].to(device)
            pred = model(image, depth)
            pred = F.interpolate(pred, (h, w), mode='bilinear',
                align_corners=False)
            pred = torch.argmax(pred, dim=1)
            pred = pred.cpu().numpy().squeeze().astype(np.uint8)
            pred_colored = dataset.color_label(pred, with_void=False)
            fig, axs = plt.subplots(1, 1, figsize=(6.4, 4.8))
            axs.imshow(pred_colored, cmap='summer')
            pred = pred[:, (1, 2, 0)]
            axs.imshow(pred, aspect='equal')
            plt.axis('off')
            height, width = pred.shape
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0,
                wspace=0)
            plt.margins(0, 0)
            plt.xticks([]), plt.yticks([])
            plt.savefig(
                '/home/cyxiong/SGACNet/samples/feature/SUNRGBD_result_Our_B/' +
                '_' + line + '.png')
            plt.show()
        else:
            break
