def print_frames_per_scene(root_dir):
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    glob_path = osp.join(root_dir, 'dataset', 'sequences', '*')
    seq_paths = sorted(glob.glob(glob_path))
    table = []
    for seq_path in seq_paths:
        frame_paths = sorted(glob.glob(osp.join(seq_path, 'image_2', '*.png')))
        cur_split = []
        for split_name in ['train', 'val', 'test', 'train_labeled',
            'train_unlabeled1', 'train_unlabeled2']:
            if osp.basename(seq_path) in getattr(splits, split_name):
                cur_split.append(split_name)
        table.append([osp.basename(seq_path), cur_split, len(frame_paths)])
        plt.imshow(np.array(Image.open(frame_paths[0])))
        plt.title(osp.basename(seq_path))
        plt.show()
    header = ['Seq', 'Split', '# Frames']
    print_table = tabulate(table, headers=header, tablefmt='psql')
    print(print_table)
