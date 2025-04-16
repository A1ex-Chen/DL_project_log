def predict(cfg, model_2d, model_3d, dataloader, pselab_path):
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
    pselab_data_list = []
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['img'] = data_batch['img'].cuda()
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
            left_idx = 0
            for batch_ind in range(len(points_idx)):
                curr_points_idx = points_idx[batch_ind]
                assert np.all(curr_points_idx)
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx]
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx
                    ] if model_3d else None
                assert np.all(pred_label_2d >= 0)
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
