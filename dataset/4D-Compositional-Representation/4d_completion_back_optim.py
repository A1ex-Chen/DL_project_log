def back_optim(model, generator, data_loader, out_dir, device, time_value,
    t_idx, latent_size, code_std=0.1, num_iterations=500):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    id_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std).cuda()
    pose_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std).cuda()
    motion_code = torch.ones(1, latent_size).normal_(mean=0, std=code_std
        ).cuda()
    id_code.requires_grad = True
    pose_code.requires_grad = True
    motion_code.requires_grad = True
    optimizer = optim.Adam([id_code, pose_code, motion_code], lr=0.03)
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
                p = pts_iou[:, t_idx, :, :].to(device)
                occ_gt = occ_iou[:, t_idx, :].to(device)
                c_i = id_code.unsqueeze(0).repeat(1, n_steps, 1)
                c_p_at_t = model.transform_to_t_eval(pts_iou_t, p=pose_code,
                    c_t=motion_code)
                c_s_at_t = torch.cat([c_i, c_p_at_t], -1)
                c_s_at_t = c_s_at_t.view(batch_size * n_steps, c_s_at_t.
                    shape[-1])
                p = p.view(batch_size * n_steps, n_pts, -1)
                occ_gt = occ_gt.view(batch_size * n_steps, n_pts)
                logits_pred = model.decode(p, c=c_s_at_t).logits
                loss_recons = F.binary_cross_entropy_with_logits(logits_pred,
                    occ_gt.view(n_steps, -1), reduction='none')
                loss_recons = loss_recons.mean()
                loss = loss_recons
                loss.backward()
                steps.set_postfix(Loss=loss.item())
                optimizer.step()
                lr_sche.step()
                if iters % 100 == 0:
                    out = generator.generate_for_completion(id_code,
                        pose_code, motion_code)
                    try:
                        mesh, stats_dict = out
                    except TypeError:
                        mesh, stats_dict = out, {}
                    modelname = model_dict['model']
                    start_idx = model_dict.get('start_idx', 0)
                    print('Saving meshes...')
                    generator.export(mesh, out_dir, modelname, start_idx)
                    print('Saving latent vectors...')
                    torch.save({'it': iters, 'id_code': id_code,
                        'pose_code': pose_code, 'motion_code': motion_code,
                        'Observations': t_idx}, os.path.join(out_dir, 
                        'latent_vecs_%d.pt' % start_idx))
