def gen_voc07_12(voc_path):
    """
    Generate voc07+12 setting dataset:
    train: # train images 16551 images
        - images/train2012
        - images/train2007
        - images/val2012
        - images/val2007
    val: # val images (relative to 'path')  4952 images
        - images/test2007
    """
    dataset_root = os.path.join(voc_path, 'voc_07_12')
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)
    dataset_settings = {'train': ['train2007', 'val2007', 'train2012',
        'val2012'], 'val': ['test2007']}
    for item in ['images', 'labels']:
        for data_type, data_list in dataset_settings.items():
            for data_name in data_list:
                ori_path = os.path.join(voc_path, item, data_name)
                new_path = os.path.join(dataset_root, item, data_type)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                print(f'[INFO]: Copying {ori_path} to {new_path}')
                for file in os.listdir(ori_path):
                    shutil.copy(os.path.join(ori_path, file), new_path)
