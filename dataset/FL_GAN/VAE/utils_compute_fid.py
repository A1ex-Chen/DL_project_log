def compute_fid(images, recon_images, device):
    torch_size = torchvision.transforms.Resize([299, 299])
    images_resize = torch_size(images).to(device)
    fid_model = torchvision.models.inception_v3(weights=torchvision.models.
        Inception_V3_Weights.DEFAULT)
    fid_model.to(device)
    global features_out_hook
    features_out_hook = []
    hook1 = fid_model.dropout.register_forward_hook(layer_hook)
    fid_model.eval()
    fid_model(images_resize)
    images_features = np.array(features_out_hook).squeeze()
    hook1.remove()
    features_out_hook = []
    recon_images_resize = torch_size(recon_images).to(device)
    hook2 = fid_model.dropout.register_forward_hook(layer_hook)
    fid_model.eval()
    fid_model(recon_images_resize)
    recon_images_features = np.array(features_out_hook).squeeze()
    hook2.remove()
    fid = calculate_fid(images_features, recon_images_features)
    return fid
