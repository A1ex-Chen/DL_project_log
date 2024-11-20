def validation(args, test_images, test_labels):
    epochs = args.epochs
    DEVICE = args.device
    DATASET = args.dataset
    CLIENT = args.client
    DP = args.dp
    K = args.num_embeddings
    D = args.embedding_dim
    if args.dataset == 'mnist' or args.dataset == 'fashion-mnist':
        N, H, W = test_images.data.shape
        C = 1
    else:
        N, C, H, W = test_images.data.shape
    model = Improved_VQVAE(K=K, D=D, channel=C, device=DEVICE).to(DEVICE)
    torch_size = torchvision.transforms.Resize([299, 299])
    re_val_images = test_images
    if DATASET == 'mnist' or DATASET == 'fashion-mnist':
        re_val_images = torch.repeat_interleave(test_images, repeats=3, dim=1)
    images_resize = torch_size(re_val_images).to(DEVICE)
    fid_model = torchvision.models.inception_v3(weights=torchvision.models.
        Inception_V3_Weights.DEFAULT)
    fid_model.to(DEVICE)
    fid_model.zero_grad()
    fid_model.eval()
    fid = []
    info = {'dataset': DATASET, 'client': CLIENT, 'dp': DP}
    weights_save_path = (
        f"Results/{info['dataset']}/{info['client']}/{info['dp']}/weights")
    if not os.path.exists(weights_save_path):
        os.makedirs(weights_save_path)
    model.zero_grad()
    model.eval()
    for e in tqdm.tqdm(range(epochs)):
        info['current_epoch'] = e + 1
        vq_vae_PATH = (weights_save_path +
            f"/vqvae/weights_central_{info['current_epoch']}.pt")
        model.load_state_dict(torch.load(vq_vae_PATH, map_location=torch.
            device(DEVICE)))
        recon_images = model(test_images.to(DEVICE))[0]
        if DATASET == 'mnist' or DATASET == 'fashion-mnist':
            recon_repeat_images = torch.repeat_interleave(recon_images,
                repeats=3, dim=1)
        else:
            recon_repeat_images = recon_images
        global features_out_hook
        features_out_hook = []
        hook1 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(images_resize)
        images_features = np.array(features_out_hook).squeeze()
        hook1.remove()
        features_out_hook = []
        recon_images_resize = torch_size(recon_repeat_images).to(DEVICE)
        hook2 = fid_model.dropout.register_forward_hook(layer_hook)
        fid_model.eval()
        fid_model(recon_images_resize)
        recon_images_features = np.array(features_out_hook).squeeze()
        hook2.remove()
        fid.append(calculate_fid(images_features, recon_images_features))
    print('Save central FID')
    metrics_save_pth = f'Results/{DATASET}/{CLIENT}/{DP}/metrics'
    if not os.path.exists(metrics_save_pth):
        os.makedirs(metrics_save_pth)
    fid_save_path = metrics_save_pth + f'/values_central_{epochs}.npy'
    np.save(fid_save_path, fid)
