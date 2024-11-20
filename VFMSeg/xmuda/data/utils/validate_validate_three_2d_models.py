def validate_three_2d_models(cfg, model_2d1, model_2d2, model_2d3,
    dataloader, val_metric_logger, pselab_path=None):
    logger = logging.getLogger('xmuda.validate')
    logger.info('Validation')
    if model_2d1 is None or model_2d2 is None or model_2d3 is None:
        raise ValueError('All three models must be valid.')
    class_names = dataloader.dataset.class_names
    evaluator_2d1 = Evaluator(class_names)
    evaluator_2d2 = Evaluator(class_names)
    evaluator_2d3 = Evaluator(class_names)
    evaluator_ensemble = Evaluator(class_names)
    pselab_data_list = []
    end = time.time()
    with torch.no_grad():
        for iteration, data_batch in enumerate(dataloader):
            data_time = time.time() - end
            if 'SCN' in cfg.DATASET_TARGET.TYPE:
                data_batch['x'][1] = data_batch['x'][1].cuda()
                data_batch['seg_label'] = data_batch['seg_label'].cuda()
                data_batch['img'] = data_batch['img'].cuda()
            else:
                raise NotImplementedError
            preds_2d1 = model_2d1(data_batch)
            preds_2d2 = model_2d2(data_batch)
            preds_2d3 = model_2d3(data_batch)
            pred_label_voxel_2d1 = preds_2d1['seg_logit'].argmax(1).cpu(
                ).numpy()
            pred_label_voxel_2d2 = preds_2d2['seg_logit'].argmax(1).cpu(
                ).numpy()
            pred_label_voxel_2d3 = preds_2d3['seg_logit'].argmax(1).cpu(
                ).numpy()
            probs_2d1 = F.softmax(preds_2d1['seg_logit'], dim=1)
            probs_2d2 = F.softmax(preds_2d2['seg_logit'], dim=1)
            probs_2d3 = F.softmax(preds_2d3['seg_logit'], dim=1)
            pred_label_voxel_ensemble = (probs_2d1 + probs_2d2 + probs_2d3
                ).argmax(1).cpu().numpy()
            seg_label = data_batch['orig_seg_label']
            points_idx = data_batch['orig_points_idx']
            left_idx = 0
            for batch_ind in range(len(seg_label)):
                curr_points_idx = points_idx[batch_ind]
                assert np.all(curr_points_idx)
                curr_seg_label = seg_label[batch_ind]
                right_idx = left_idx + curr_points_idx.sum()
                pred_label_2d1 = pred_label_voxel_2d1[left_idx:right_idx]
                pred_label_2d2 = pred_label_voxel_2d2[left_idx:right_idx]
                pred_label_2d3 = pred_label_voxel_2d3[left_idx:right_idx]
                pred_label_ensemble = pred_label_voxel_ensemble[left_idx:
                    right_idx]
                evaluator_2d1.update(pred_label_2d1, curr_seg_label)
                evaluator_2d2.update(pred_label_2d2, curr_seg_label)
                evaluator_2d3.update(pred_label_2d3, curr_seg_label)
                evaluator_ensemble.update(pred_label_ensemble, curr_seg_label)
                if pselab_path is not None:
                    assert np.all(pred_label_2d1 >= 0)
                    assert np.all(pred_label_2d2 >= 0)
                    assert np.all(pred_label_2d3 >= 0)
                    curr_probs_2d1 = probs_2d1[left_idx:right_idx]
                    curr_probs_2d2 = probs_2d2[left_idx:right_idx]
                    curr_probs_2d3 = probs_2d3[left_idx:right_idx]
                    current_probs_ensemble = (curr_probs_2d1 +
                        curr_probs_2d2 + curr_probs_2d3) / 3
                    pselab_data_list.append({'probs_2d':
                        current_probs_ensemble[range(len(
                        pred_label_ensemble)), pred_label_ensemble].cpu().
                        numpy(), 'pseudo_label_2d': pred_label_ensemble.
                        astype(np.uint8), 'probs_3d': None,
                        'pseudo_label_3d': None})
                left_idx = right_idx
            seg_loss_2d1 = F.cross_entropy(preds_2d1['seg_logit'],
                data_batch['seg_label'])
            seg_loss_2d2 = F.cross_entropy(preds_2d2['seg_logit'],
                data_batch['seg_label'])
            seg_loss_2d3 = F.cross_entropy(preds_2d3['seg_logit'],
                data_batch['seg_label'])
            val_metric_logger.update(seg_loss_2d1=seg_loss_2d1)
            val_metric_logger.update(seg_loss_2d2=seg_loss_2d2)
            val_metric_logger.update(seg_loss_2d3=seg_loss_2d3)
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
        val_metric_logger.update(seg_iou_2d1=evaluator_2d1.overall_iou)
        eval_list.append(('2D_1', evaluator_2d1))
        val_metric_logger.update(seg_iou_2d2=evaluator_2d2.overall_iou)
        eval_list.append(('2D_2', evaluator_2d2))
        val_metric_logger.update(seg_iou_2d3=evaluator_2d3.overall_iou)
        eval_list.append(('2D_3', evaluator_2d3))
        eval_list.append(('2D+3D', evaluator_ensemble))
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
