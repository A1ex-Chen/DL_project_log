def parse_args():
    parser = argparse.ArgumentParser(description='Bert Mimic Synth')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size'
        )
    parser.add_argument('--num_epochs', type=int, default=10, help=
        'Adam learning rate')
    parser.add_argument('--learning_rate', type=float, default=0.001, help=
        'Adam learning rate')
    parser.add_argument('--eps', type=float, default=1e-07, help='Adam epsilon'
        )
    parser.add_argument('--num_train_samples', type=int, default=10000,
        help='Number of training samples')
    parser.add_argument('--num_valid_samples', type=int, default=10000,
        help='Number of valid samples')
    parser.add_argument('--num_test_samples', type=int, default=10000, help
        ='Number of test samples')
    parser.add_argument('--num_classes', type=int, default=10, help=
        'Number of clases')
    parser.add_argument('--weight_decay', type=float, default=0.0, help=
        'weight decay')
    parser.add_argument('--device', type=str, default='cuda', help=
        'path to the model weights')
    parser.add_argument('--pretrained_weights_path', type=str, help=
        'path to the model weights')
    return parser.parse_args()
