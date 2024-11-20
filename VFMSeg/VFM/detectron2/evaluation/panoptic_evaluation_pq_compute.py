def pq_compute(gt_json_file, pred_json_file, gt_folder=None, pred_folder=None):
    start_time = time.time()
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    with open(pred_json_file, 'r') as f:
        pred_json = json.load(f)
    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}
    print('Evaluation panoptic segmentation metrics:')
    print('Ground truth:')
    print('\tSegmentation folder: {}'.format(gt_folder))
    print('\tJSON file: {}'.format(gt_json_file))
    print('Prediction:')
    print('\tSegmentation folder: {}'.format(pred_folder))
    print('\tJSON file: {}'.format(pred_json_file))
    if not os.path.isdir(gt_folder):
        raise Exception(
            "Folder {} with ground truth segmentations doesn't exist".
            format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist"
            .format(pred_folder))
    pred_annotations = {el['image_id']: el for el in pred_json['annotations']}
    matched_annotations_list = []
    n_missing_files = 0
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            n_missing_files += 1
            continue
            raise Exception('no prediction for the image with id: {}'.
                format(image_id))
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))
    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder,
        pred_folder, categories)
    metrics = [('All', None), ('Things', True), ('Stuff', False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories,
            isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print('{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}'.format('', 'PQ', 'SQ',
        'RQ', 'N'))
    print('-' * (10 + 7 * 4))
    for name, _isthing in metrics:
        print('{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}'.format(name, 100 *
            results[name]['pq'], 100 * results[name]['sq'], 100 * results[
            name]['rq'], results[name]['n']))
    t_delta = time.time() - start_time
    print('Time elapsed: {:0.2f} seconds'.format(t_delta))
    print(f'Missing files: {n_missing_files}')
    return results
