def predict(cfg, args, model_2d, model_3d, dataloader, pselab_path,
    save_ensemble=False):
    """
    Function to save pseudo labels. The difference with the "validate" function is that no ground truth labels are
    required, i.e. no evaluation (mIoU) is computed.

    :param cfg: Configuration node.
    :param model_2d: 2D model.
    :param model_3d: 3D model. Optional.
    :param dataloader: Dataloader for the test dataset.
    :param pselab_path: Path to dictionary where to save the pseudo labels as npy file.
    :param save_ensemble: Whether to save the ensemble labels (2D+3D).
    """
    logger = logging.getLogger('xmuda.predict')
    logger.info('Prediction of Pseudo Labels')
    if not pselab_path:
        raise ValueError(
            'A pseudo label path must be provided for this function.')
    if not model_2d:
        raise ValueError('A 2D model must be provided.')
    if save_ensemble and not model_3d:
        raise ValueError('For ensembling, a 3D model needs to be provided.')
    pselab_data_list = []
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with_vfm = False
    if (args.vfmlab == True and args.vfm_pth is not None and args.vfm_cfg
         is not None):
        with_vfm = True
        if ('SemanticKITTISCN' == cfg.DATASET_TARGET.TYPE and 10 == cfg.
            MODEL_2D.NUM_CLASSES):
            mapping = 'A2D2SCN'
        elif 'SemanticKITTISCN' == cfg.DATASET_TARGET.TYPE and 6 == cfg.MODEL_2D.NUM_CLASSES:
            mapping = 'SemanticKITTISCN'
        elif 'NuScenesLidarSegSCN' == cfg.DATASET_TARGET.TYPE and 6 == cfg.MODEL_2D.NUM_CLASSES:
            mapping = 'NuScenesLidarSegSCN'
        else:
            raise ValueError('Unsupported type of Label Mapping: {}.'.
                format(cfg.DATASET_TARGET.TYPE))
        model_SEEM = build_SEEM(args.vfm_pth, args.vfm_cfg, cuda_device_idx)
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda(device=
                    cuda_device_idx)
                data_batch['img'] = data_batch['img'].cuda(device=
                    cuda_device_idx)
            else:
                raise NotImplementedError
            preds_2d = model_2d(data_batch)
            preds_3d = model_3d(data_batch) if model_3d else None
            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy()
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy(
                ) if model_3d else None
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1)
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1
                ) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu(
                ).numpy() if model_3d else None
            points_idx = data_batch['orig_points_idx']
            if with_vfm:
                img_indices_orig = data_batch['img_indices_orig']
                img_paths_orig = data_batch['img_paths']
            left_idx = 0
            for batch_ind in range(len(points_idx)):
                curr_points_idx = points_idx[batch_ind]
                assert np.all(curr_points_idx)
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx
                    ] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:
                    right_idx] if model_3d else None
                assert np.all(pred_label_2d >= 0)
                if with_vfm:
                    preds_logits_SEEM, _ = call_SEEM(model_SEEM,
                        img_paths_orig[batch_ind], mapping)
                    preds_logits_SEEM = preds_logits_SEEM.permute(1, 2, 0)[
                        img_indices_orig[batch_ind][:, 0], img_indices_orig
                        [batch_ind][:, 1]]
                    sf = F.softmax(preds_logits_SEEM, dim=1)
                    pred_label_2d_vfm_ensemble = probs_2d[left_idx:right_idx
                        ] + sf
                    curr_probs_2d = pred_label_2d_vfm_ensemble / 2
                    pred_label_2d = pred_label_2d_vfm_ensemble.argmax(1).cpu(
                        ).numpy()
                    assert np.all(pred_label_2d >= 0)
                else:
                    curr_probs_2d = probs_2d[left_idx:right_idx]
                curr_probs_3d = probs_3d[left_idx:right_idx
                    ] if model_3d else None
                pseudo_label_dict = {'probs_2d': curr_probs_2d[range(len(
                    pred_label_2d)), pred_label_2d].cpu().numpy(),
                    'pseudo_label_2d': pred_label_2d.astype(np.uint8),
                    'probs_3d': curr_probs_3d[range(len(pred_label_3d)),
                    pred_label_3d].cpu().numpy() if model_3d else None,
                    'pseudo_label_3d': pred_label_3d.astype(np.uint8) if
                    model_3d else None}
                if save_ensemble:
                    pseudo_label_dict['pseudo_label_ensemble'
                        ] = pred_label_ensemble.astype(np.uint8)
                pselab_data_list.append(pseudo_label_dict)
                left_idx = right_idx
            cur_iter = iteration + 1
            if (cur_iter == 1 or cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.
                VAL.LOG_PERIOD == 0):
                logger.info('  '.join(['iter: {iter}/{total_iter}',
                    'max mem: {memory:.0f}']).format(iter=cur_iter,
                    total_iter=len(dataloader), memory=torch.cuda.
                    max_memory_allocated() / 1024.0 ** 2))
        np.save(pselab_path, pselab_data_list)
        logger.info('Saved pseudo label data to {}'.format(pselab_path))
