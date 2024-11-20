def parse_args():
    parser = argparse.ArgumentParser(description='Encoder ImageNet Training')
    parser.add_argument('--data_dir', help='path to ImageNet data')
    parser.add_argument('--results_dir', default='./trained_models/imagenet')
    parser.add_argument('--epochs', default=90, type=int, metavar='N', help
        ='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=256, type=int,
        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight_decay', '--wd', default=0.0001, type=
        float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
        help='momentum')
    parser.add_argument('--encoder', type=str, default='mobilenet_v2',
        choices=['mobilenet_v2'], help='Wich mobilenetv2 to train')
    parser.add_argument('-p', '--print-freq', default=1, type=int, metavar=
        'N', help='print frequency (default: 1)')
    parser.add_argument('--finetune', default=False, action='store_true',
        help=
        'Set this if you have pretrained weights that only need to be adapted')
    parser.add_argument('--weight_file', type=str, help=
        'path to weight file for finetuning')
    args = parser.parse_args()
    if args.finetune:
        args.lr = 0.001
    args.lr = args.lr * args.batch_size / 256
    return args
