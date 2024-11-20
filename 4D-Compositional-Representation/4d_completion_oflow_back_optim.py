def back_optim(model, generator, data_loader, out_dir, device, time_value,
    t_idx, latent_size, code_std=0.1, num_iterations=500):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    shape_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std).cuda(
        )
    motion_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std
        ).cuda()
    shape_code.requires_grad = True
    motion_code.requires_grad = True
    optimizer = optim.Adam([shape_code, motion_code], lr=0.03)
    lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    print('Seen Frames:')
    print(t_idx)
    with trange(num_iterations, ncols=80) as steps:
        iters = 0
        for _ in steps:
            for batch in data_loader:
                iters += 1
                idx = batch['idx'].item()
                try:
                    model_dict = dataset.get_model_dict(idx)
                except AttributeError:
                    model_dict = {'model': str(idx), 'category': 'n/a'}
                model.eval()
                optimizer.zero_grad()
                pts_iou = batch.get('points')
                occ_iou = batch.get('points.occ')
                pts_iou_t = torch.from_numpy(time_value).to(device)
                batch_size, _, n_pts, dim = pts_iou.shape
                n_steps = pts_iou_t.shape[0]
                z_t = torch.empty(batch_size, 0).to(device)
                p = pts_iou[:, t_idx, :, :].to(device)
                occ_gt = occ_iou[:, t_idx, :].to(device)
                c_t = motion_code.unsqueeze(1).repeat(1, n_steps, 1).view(
                    batch_size * n_steps, -1)
                z_t = z_t.unsqueeze(1).repeat(1, n_steps, 1).view(
                    batch_size * n_steps, -1)
                pts_iou_t0 = model.transform_to_t0(pts_iou_t, p[0], z_t, c_t)
                pts_iou_t0 = pts_iou_t0.view(batch_size * n_steps, n_pts, dim)
                c_s = shape_code
                z = torch.empty(batch_size, 0).to(device)
                c_s = c_s.unsqueeze(1).repeat(1, n_steps, 1).view(
                    batch_size * n_steps, -1)
                z = z.unsqueeze(1).repeat(1, n_steps, 1).view(batch_size *
                    n_steps, -1)
                logits_pred = model.decode(pts_iou_t0, z, c_s).logits
                loss_recons = F.binary_cross_entropy_with_logits(logits_pred,
                    occ_gt.view(n_steps, -1), reduction='none')
                loss_recons = loss_recons.mean()
                loss = loss_recons
                loss.backward()
                steps.set_postfix(Loss=loss.item())
                optimizer.step()
                lr_sche.step()
                if iters % 100 == 0:
                    out = generator.generate_for_completion(shape_code,
                        motion_code)
                    try:
                        mesh, stats_dict = out
                    except TypeError:
                        mesh, stats_dict = out, {}
                    modelname = model_dict['model']
                    start_idx = model_dict.get('start_idx', 0)
                    print('Saving meshes...')
                    generator.export(mesh, out_dir, modelname, start_idx)
                    print('Saving latent vectors...')
                    torch.save({'it': iters, 'shape_code': shape_code,
                        'motion_code': motion_code, 'Observations': t_idx},
                        os.path.join(out_dir, 'latent_vecs_%d.pt' % start_idx))
