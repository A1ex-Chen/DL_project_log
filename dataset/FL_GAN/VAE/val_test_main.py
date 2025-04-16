def main(args, trainset, testset):
    """Train the network on the training set."""
    DATASET = args.dataset
    DP = args.dp
    CLIENT = args.client
    trainset_len = len(trainset)
    bound = int(math.floor(trainset_len * args.split_rate))
    train_set = torch.utils.data.Subset(trainset, range(bound))
    validation_set = torch.utils.data.Subset(trainset, range(bound,
        trainset_len))
    metrics = train_vq_vae_with_gated_pixelcnn_prior(args, train_set,
        validation_set, testset)
    metrics_save_pth = f'Results/{DATASET}/{CLIENT}/{DP}/metrics'
    if not os.path.exists(metrics_save_pth):
        os.makedirs(metrics_save_pth)
    vqvqe_train_loss_save_pth = (metrics_save_pth +
        f'/metrics_dict_{args.epochs}.npy')
    np.save(vqvqe_train_loss_save_pth, metrics)
