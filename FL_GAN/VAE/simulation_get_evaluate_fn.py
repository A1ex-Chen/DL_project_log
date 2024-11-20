def get_evaluate_fn(args):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config:
        Dict[str, Scalar]):
        my_device = ''
        try:
            if torch.backends.mps.is_built():
                my_device = 'mps'
        except AttributeError:
            if torch.cuda.is_available():
                my_device = 'cuda:0'
            else:
                my_device = 'cpu'
        DATASET = args.dataset
        CLIENT = args.client
        DP = args.dp
        model = VAE(DATASET)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(my_device)
        _, testset = load_partition(CLIENT, DATASET)
        valLoader = DataLoader(testset, batch_size=args.batch_size, shuffle
            =True)
        images, labels = next(iter(valLoader))
        recon_images, _, _ = model(images.to(my_device))
        ground_truth = images[:16]
        predictions = recon_images[:16]
        fig = plt.figure(figsize=(3, 3))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.transpose(predictions[i, :, :].cpu().detach().numpy(),
                axes=[1, 2, 0])
            plt.imshow(img)
            plt.axis('off')
        image_save_path = f'Results/{DATASET}/{CLIENT}/{DP}/FL_G_data'
        if not os.path.exists(image_save_path):
            os.makedirss(image_save_path)
        g_images_save_path = (image_save_path +
            f'/image_at_epoch_{server_round:03d}_G.png')
        plt.savefig(g_images_save_path)
        plt.close()
        fig2 = plt.figure(figsize=(3, 3))
        for i in range(ground_truth.shape[0]):
            plt.subplot(4, 4, i + 1)
            img = np.transpose(ground_truth[i, :, :].cpu().detach().numpy(),
                axes=[1, 2, 0])
            plt.imshow(img)
            plt.axis('off')
        r_images_save_pth = (image_save_path +
            f'/image_at_epoch_{server_round:03d}.png')
        plt.savefig(r_images_save_pth)
        plt.close()
        label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog',
            'horse', 'monkey', 'ship', 'truck']
        val_labels = [label_names[i] for i in np.array(labels)]
        labels_save_path = f'Results/{DATASET}/{CLIENT}/{DP}/Labels'
        if not os.path.exists(labels_save_path):
            os.makedirss(labels_save_path)
        labels_save_path += f'/label_at_epoch_{server_round:03d}.txt'
        np.savetxt(labels_save_path, labels, fmt='%s')
        my_device = ''
        try:
            if torch.backends.mps.is_built():
                my_device = 'mps'
        except AttributeError:
            if torch.cuda.is_available():
                my_device = 'cuda:0'
            else:
                my_device = 'cpu'
        model.to(my_device)
        print(f'Global validation starting with device: {my_device}...')
        loss, fid = test(model, valLoader, device=my_device)
        print(f'Global validation end with FID: {fid}')
        return loss, {'fid': float(fid)}
    return evaluate
