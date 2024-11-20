def validate(cfg, args, model_2d, model_3d, dataloader, val_metric_logger,
    pselab_path=None, model_ensemble=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')
    dual_model = model_2d is not None and model_3d is not None
    class_names = dataloader.dataset.class_names
    evaluator_2d = Evaluator(class_names) if model_2d else None
    evaluator_3d = Evaluator(class_names) if model_3d else None
    evaluator_ensemble = Evaluator(class_names) if dual_model else None
    evaluator_VFM_ensemble = Evaluator(class_names) if model_ensemble else None
    pselab_data_list = []
    with_vfm = False
    if (args is not None and args.vfmlab == True and args.vfm_pth is not
        None and args.vfm_cfg is not None):
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
        evaluator_vfm = Evaluator(class_names)
        evaluator_2d_vfm_ensemble = Evaluator(class_names)
        evaluator_2d_vfm_3d_ensemble = Evaluator(class_names)
        evaluator_vfm_seg_mask = Evaluator(class_names)
    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda(device=
                    cuda_device_idx)
                data_batch['seg_label'] = data_batch['seg_label'].cuda(device
                    =cuda_device_idx)
                data_batch['img'] = data_batch['img'].cuda(device=
                    cuda_device_idx)
                if with_vfm:
                    images_orig = data_batch['img_instances_orig']
                    images_indices = data_batch['img_indices']
            else:
                raise NotImplementedError
            preds_2d = model_2d(data_batch) if model_2d else None
            preds_3d = model_3d(data_batch) if model_3d else None
            pred_label_voxel_2d = preds_2d['seg_logit'].argmax(1).cpu().numpy(
                ) if model_2d else None
            pred_label_voxel_3d = preds_3d['seg_logit'].argmax(1).cpu().numpy(
                ) if model_3d else None
            probs_2d = F.softmax(preds_2d['seg_logit'], dim=1
                ) if model_2d else None
            probs_3d = F.softmax(preds_3d['seg_logit'], dim=1
                ) if model_3d else None
            pred_label_voxel_ensemble = (probs_2d + probs_3d).argmax(1).cpu(
                ).numpy() if dual_model else None
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            if with_vfm:
                img_indices_orig = data_batch['img_indices_orig']
                img_paths_orig = data_batch['img_paths']
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                assert np.all(curr_points_idx)
                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                if with_vfm:
                    preds_logits_SEEM, pan_seg_SEEM = call_SEEM(model_SEEM,
                        img_paths_orig[batch_ind], mapping)
                    preds_logits_SEEM = preds_logits_SEEM.permute(1, 2, 0)[
                        img_indices_orig[batch_ind][:, 0], img_indices_orig
                        [batch_ind][:, 1]]
                    pan_seg_SEEM = pan_seg_SEEM[img_indices_orig[batch_ind]
                        [:, 0], img_indices_orig[batch_ind][:, 1]]
                    pred_label_vfm = preds_logits_SEEM.argmax(1).cpu().numpy(
                        ) if model_2d else None
                    pred_label_vfm_seg_mask = pan_seg_SEEM.cpu().numpy(
                        ) if model_2d else None
                    sf = F.softmax(preds_logits_SEEM, dim=1)
                    pred_label_2d_vfm_ensemble = probs_2d[left_idx:right_idx
                        ] + sf
                    pred_label_2d_vfm_3d_ensemble = (
                        pred_label_2d_vfm_ensemble / 2 + probs_3d[left_idx:
                        right_idx]).argmax(1).cpu().numpy()
                    pred_label_2d_vfm_ensemble = (pred_label_2d_vfm_ensemble
                        .argmax(1).cpu().numpy())
                pred_label_2d = pred_label_voxel_2d[left_idx:right_idx
                    ] if model_2d else None
                pred_label_3d = pred_label_voxel_3d[left_idx:right_idx
                    ] if model_3d else None
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:
                    right_idx] if dual_model else None
                if model_2d:
                    evaluator_2d.update(pred_label_2d, curr_seg_label)
                if model_3d:
                    evaluator_3d.update(pred_label_3d, curr_seg_label)
                if dual_model:
                    evaluator_ensemble.update(pred_label_ensemble,
                        curr_seg_label)
                if with_vfm:
                    evaluator_vfm.update(pred_label_vfm, curr_seg_label)
                    evaluator_2d_vfm_ensemble.update(pred_label_2d_vfm_ensemble
                        , curr_seg_label)
                    evaluator_2d_vfm_3d_ensemble.update(
                        pred_label_2d_vfm_3d_ensemble, curr_seg_label)
                    evaluator_vfm_seg_mask.update(pred_label_vfm_seg_mask,
                        curr_seg_label)
                if pselab_path is not None:
                    if model_2d:
                        assert np.all(pred_label_2d >= 0)
                    if model_3d:
                        assert np.all(pred_label_3d >= 0)
                    curr_probs_2d = probs_2d[left_idx:right_idx
                        ] if model_2d else None
                    curr_probs_3d = probs_3d[left_idx:right_idx
                        ] if model_3d else None
                    pselab_data_list.append({'probs_2d': curr_probs_2d[
                        range(len(pred_label_2d)), pred_label_2d].cpu().
                        numpy() if model_2d else None, 'pseudo_label_2d': 
                        pred_label_2d.astype(np.uint8) if model_2d else
                        None, 'probs_3d': curr_probs_3d[range(len(
                        pred_label_3d)), pred_label_3d].cpu().numpy() if
                        model_3d else None, 'pseudo_label_3d': 
                        pred_label_3d.astype(np.uint8) if model_3d else None})
                left_idx = right_idx
            seg_loss_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch
                ['seg_label']) if model_2d else None
            seg_loss_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch
                ['seg_label']) if model_3d else None
            if seg_loss_2d is not None:
                val_metric_logger.update(seg_loss_2d=seg_loss_2d)
            if seg_loss_3d is not None:
                val_metric_logger.update(seg_loss_3d=seg_loss_3d)
            batch_time = time.time() - end
            val_metric_logger.update(time=batch_time, data=data_time)
            end = time.time()
            cur_iter = iteration + 1
            if (cur_iter == 1 or cfg.VAL.LOG_PERIOD > 0 and cur_iter % cfg.
                VAL.LOG_PERIOD == 0):
                logger.info(val_metric_logger.delimiter.join([
                    'iter: {iter}/{total_iter}', '{meters}',
                    'max mem: {memory:.0f}']).format(iter=cur_iter,
                    total_iter=len(dataloader), meters=str(
                    val_metric_logger), memory=torch.cuda.
                    max_memory_allocated() / 1024.0 ** 2))
        eval_list = []
        if evaluator_2d is not None:
            val_metric_logger.update(seg_iou_2d=evaluator_2d.overall_iou)
            eval_list.append(('2D', evaluator_2d))
        if evaluator_3d is not None:
            val_metric_logger.update(seg_iou_3d=evaluator_3d.overall_iou)
            eval_list.append(('3D', evaluator_3d))
        if dual_model:
            eval_list.append(('2D+3D', evaluator_ensemble))
        if with_vfm:
            eval_list.append(('2D_VFM_Logits', evaluator_vfm))
            eval_list.append(('2D_VFM_Seg_Mask', evaluator_vfm_seg_mask))
            eval_list.append(('2D_VFM_ensemble', evaluator_2d_vfm_ensemble))
            eval_list.append(('2D_VFM_3d_ensemble',
                evaluator_2d_vfm_3d_ensemble))
        for modality, evaluator in eval_list:
            logger.info('{} overall accuracy: {:.2f}%'.format(modality, 
                100.0 * evaluator.overall_acc))
            logger.info('{} overall IOU: {:.2f}'.format(modality, 100.0 *
                evaluator.overall_iou))
            logger.info('{} class-wise segmentation accuracy and IoU.\n{}'.
                format(modality, evaluator.print_table()))
        if pselab_path is not None:
            np.save(pselab_path, pselab_data_list)
            logger.info('Saved pseudo label data to {}'.format(pselab_path))
