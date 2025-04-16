def compute_class_weights():
    preprocess_dir = (
        '/datasets_local/datasets_mjaritz/semantic_kitti_preprocess/preprocess'
        )
    split = 'train',
    dataset = SemanticKITTIBase(split, preprocess_dir, merge_classes_style=
        'A2D2')
    num_classes = len(dataset.class_names)
    points_per_class = np.zeros(num_classes, int)
    for i, data in enumerate(dataset.data):
        print('{}/{}'.format(i, len(dataset)))
        labels = dataset.label_mapping[data['seg_labels']]
        points_per_class += np.bincount(labels[labels != -100], minlength=
            num_classes)
    class_weights = np.log(5 * points_per_class.sum() / points_per_class)
    print('log smoothed class weights: ', class_weights / class_weights.min())
