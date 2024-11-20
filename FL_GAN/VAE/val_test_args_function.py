def args_function():
    parser = argparse.ArgumentParser(description='Opacus Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--grad_sample_mode', type=str, default='hooks')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
        metavar='B', help='Batch size')
    parser.add_argument('--num_reconstruction', type=int, default=36, help=
        'number for reconstruction')
    parser.add_argument('--num_sampling', type=int, default=36, help=
        'number of samplings')
    parser.add_argument('-n', '--epochs', type=int, default=3, metavar='N',
        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help=
        'beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help=
        'beta2 for adam. default=0.999')
    parser.add_argument('--weight_decay', type=float, default=1e-05,
        metavar='WD', help='weight decay')
    parser.add_argument('--sigma', type=float, default=1.0, metavar='S',
        help='Noise multiplier')
    parser.add_argument('-c', '--max_per_sample_grad_norm', type=float,
        default=1.0, metavar='C', help='Clip per-sample gradients to this norm'
        )
    parser.add_argument('--epsilon', type=float, default=10, help=
        'Target Epsilon')
    parser.add_argument('--delta', type=float, default=1e-05, metavar='D',
        help='Target delta')
    parser.add_argument('--device', type=str, default='cpu', help=
        'default GPU ID for model')
    parser.add_argument('--dataset', type=str, default='cifar', help=
        'mnist, fashion-mnist, cifar, stl')
    parser.add_argument('--dp', type=str, default='gaussian', help=
        'Disable privacy training and just train with vanilla type')
    parser.add_argument('--client', type=int, default=1, help=
        'Number of clients, 1 for centralized, 2/3/4/5 for federated learning')
    parser.add_argument('--secure_rng', action='store_true', default=False,
        help=
        'Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost'
        )
    parser.add_argument('--clip_per_layer', action='store_true', default=
        False, help=
        'Use static per-layer clipping with the same clipping threshold for each layer. Necessary for DDP. If `False` (default), uses flat clipping.'
        )
    parser.add_argument('--lr_schedule', type=str, choices=['constant',
        'cos'], default='cos')
    parser.add_argument('-D', '--embedding_dim', type=int, default=64, help
        ='Embedding dimention')
    parser.add_argument('-K', '--num_embeddings', type=int, default=512,
        help='Embedding dimention')
    parser.add_argument('--split_rate', type=float, default=0.8, help=
        'Splite ratio for train set and test set')
    args = parser.parse_args()
    return args
