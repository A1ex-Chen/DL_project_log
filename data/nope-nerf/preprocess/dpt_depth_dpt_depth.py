def dpt_depth(cfg, depth_save_dir):
    torch.manual_seed(0)
    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    network_type = cfg['model']['network_type']
    if network_type == 'official':
        model = mdl.OfficialStaticNerf(cfg)
    rendering_cfg = cfg['rendering']
    renderer = mdl.Renderer(model, rendering_cfg, device=device)
    nope_nerf = mdl.get_model(renderer, cfg, device=device)
    train_loader, train_dataset = get_dataloader(cfg, mode='all', shuffle=False
        )
    nope_nerf.eval()
    DPT_model = nope_nerf.depth_estimator.to(device)
    if not os.path.exists(depth_save_dir):
        os.makedirs(depth_save_dir)
    img_list = train_dataset['img'].img_list
    for data in train_loader:
        img_normalised = data.get('img.normalised_img').to(device)
        idx = data.get('img.idx')
        img_name = img_list[idx]
        depth = DPT_model(img_normalised)
        np.savez(os.path.join(depth_save_dir, 'depth_{}.npz'.format(
            img_name.split('.')[0])), pred=depth.detach().cpu())
        depth_array = depth[0].detach().cpu().numpy()
        imageio.imwrite(os.path.join(depth_save_dir, '{}.png'.format(
            img_name.split('.')[0])), np.clip(255.0 / depth_array.max() * (
            depth_array - depth_array.min()), 0, 255).astype(np.uint8))
