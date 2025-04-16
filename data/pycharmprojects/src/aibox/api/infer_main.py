def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help
        ='path to checkpoint')
    parser.add_argument('-l', '--lower_prob_thresh', type=float, default=
        0.7, help='threshold of lower probability')
    parser.add_argument('-u', '--upper_prob_thresh', type=float, default=
        1.0, help='threshold of upper probability')
    parser.add_argument('--device_ids', type=str)
    parser.add_argument('image_pattern_list', type=str, help=
        'path to image pattern list')
    parser.add_argument('results_dir', type=str, help=
        'path to result directory')
    subparsers = parser.add_subparsers(dest='task', help='task name')
    classification_subparser = subparsers.add_parser(Task.Name.
        CLASSIFICATION.value)
    detection_subparser = subparsers.add_parser(Task.Name.DETECTION.value)
    instance_segmentation_subparser = subparsers.add_parser(Task.Name.
        INSTANCE_SEGMENTATION.value)
    args = parser.parse_args()
    path_to_checkpoint = args.checkpoint
    lower_prob_thresh = args.lower_prob_thresh
    upper_prob_thresh = args.upper_prob_thresh
    device_ids = args.device_ids
    path_to_image_pattern_list = literal_eval(args.image_pattern_list)
    path_to_results_dir = args.results_dir
    task_name = Task.Name(args.task)
    if device_ids is not None:
        device_ids = literal_eval(device_ids)
    path_to_image_list = []
    for path_to_image_pattern in path_to_image_pattern_list:
        path_to_image_list += glob.glob(path_to_image_pattern)
    path_to_image_list = sorted(path_to_image_list)
    print('Arguments:\n' + ' '.join(sys.argv[1:]))
    _infer(task_name, path_to_checkpoint, lower_prob_thresh,
        upper_prob_thresh, device_ids, path_to_image_list, path_to_results_dir)
