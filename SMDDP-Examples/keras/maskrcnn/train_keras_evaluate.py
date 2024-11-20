def evaluate(cfg, detector_model):
    eval_dataset = build_dataset(cfg, mode='eval')
    coco_prediction = detector_model.predict(x=eval_dataset)
    imgIds, box_predictions, mask_predictions = evaluation.process_prediction(
        coco_prediction)
    from smdistributed.dataparallel.tensorflow import get_worker_comm
    comm = get_worker_comm()
    imgIds_mpi_list = comm.gather(imgIds, root=0)
    box_predictions_mpi_list = comm.gather(box_predictions, root=0)
    mask_predictions_mpi_list = comm.gather(mask_predictions, root=0)
    if rank == 0:
        imgIds = []
        box_predictions = []
        mask_predictions = []
        for i in imgIds_mpi_list:
            imgIds.extend(i)
        print('Running Evaluation for {} images'.format(len(set(imgIds))))
        for i in box_predictions_mpi_list:
            box_predictions.extend(i)
        predictions = {'bbox': box_predictions}
        if cfg.MODEL.INCLUDE_MASK:
            for i in mask_predictions_mpi_list:
                mask_predictions.extend(i)
            predictions['segm'] = mask_predictions
        stat_dict = evaluation.evaluate_coco_predictions(cfg.PATHS.
            VAL_ANNOTATIONS, predictions.keys(), predictions, verbose=False)
        print(stat_dict)
