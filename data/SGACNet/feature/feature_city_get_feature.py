def get_feature():
    parser = ArgumentParserRGBDSegmentation(description=
        'Efficient RGBD Indoor Sematic Segmentation (Inference)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    parser.add_argument('--ckpt_path', type=str, required=True, help=
        'Path to the checkpoint of the trained model.')
    parser.add_argument('--depth_scale', type=float, default=1.0, help=
        'Additional depth scaling factor to apply.')
    args = parser.parse_args()
    model, device = build_model(args, n_classes=19)
    checkpoint = torch.load(args.ckpt_path, map_location=lambda storage,
        loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    print('Loaded checkpoint from {}'.format(args.ckpt_path))
    model.eval()
    model.to(device)
    root_path = '/home/yzhang/SGACNet/datasets/cityscapes/train/'
    f = open(
        '/home/yzhang/SGACNet/datasets/cityscapes/train_disparity_raw1.txt',
        'r')
    p = open('/home/yzhang/SGACNet/datasets/cityscapes/train_rgb1.txt', 'r')
    q = open('/home/yzhang/SGACNet/datasets/cityscapes/list.txt', 'r')
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
            depth_path = root_path + 'disparity_raw/' + line1
            rgb_path = root_path + 'rgb/' + line0
            img_rgb = get_picture(rgb_path, transform)
            img_rgb = img_rgb.unsqueeze(0)
            img_rgb = img_rgb.to(device)
            img_depth = get_picture(depth_path, transform)
            img_depth = img_depth.unsqueeze(0)
            img_depth = img_depth.to(device)
            exact_list = ['se_layer4', 'context_module']
            pred, all_dict = model(img_rgb, img_depth)
            outputs = []
            for item in exact_list:
                x = all_dict[item]
                outputs.append(x)
            x = outputs
            k = 0
            print(x[0].shape[1])
            for item in x:
                c = item.shape[1]
                plt.figure()
                name = exact_list[k]
                plt.suptitle(name)
                for i in range(c):
                    wid = math.ceil(math.sqrt(c))
                    ax = plt.subplot(wid, wid, i + 1)
                    ax.set_title('{}'.format(i))
                    ax.axis('off')
                    figure_map = item.data.cpu().numpy()[0, i, :, :]
                    plt.imshow(figure_map, cmap='jet')
                visualize_feature_map_sum(item, line, name)
                k = k + 1
            plt.show()
        else:
            break
    f.close()
