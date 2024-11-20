def parse_args():
    parser = argparse.ArgumentParser(description='RNAseq generator')
    parser.add_argument('-d', '--latent_dim', type=int, default=10, help=
        'latent dimensions')
    parser.add_argument('-m', '--model', default='cvae', help=
        'generator model to use: ae, vae, cvae')
    parser.add_argument('-e', '--epochs', type=int, default=20, help=
        'number of epochs to train')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help=
        'mini batch size')
    parser.add_argument('-k', '--top_k_types', type=int, default=20, help=
        'number of top sample types to use')
    parser.add_argument('-n', '--n_samples', type=int, default=10000, help=
        'number of RNAseq samples to generate')
    parser.add_argument('--plot', action='store_true', help=
        'plot test performance comparision with and without synthetic training data'
        )
    parser.add_argument('--seed', type=int, default=2021, help=
        'random seed state')
    return parser.parse_args()
