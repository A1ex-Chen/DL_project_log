def process_arguments():
    """ Collect the input arguments according to the syntax
        Return a parser with the arguments
    """
    parser = argparse.ArgumentParser(description=
        'Train new netword on a dataset and save the model')
    parser.add_argument('--data_directory', action='store', default=
        'D:\\dataset\\insect/', help='Input directory for training data')
    parser.add_argument('--save_dir', action='store', dest='save_directory',
        default='checkpoint_dir', help=
        'Directory where the checkpoint file is saved')
    parser.add_argument('--arch', action='store', dest='choosen_archi',
        default='resnet50', help='Choosen models to train chicken')
    parser.add_argument('--learning_rate', action='store', dest=
        'learning_rate', type=float, default=0.0006, help=
        'Neural Network learning rate')
    parser.add_argument('--hidden_units', action='store', dest=
        'hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', action='store', dest='epochs', type=int,
        default=12, help='Number of Epochs for the training')
    parser.add_argument('--gpu', action='store_true', default=True, help=
        'Use GPU. The default is CPU')
    return parser.parse_args()
