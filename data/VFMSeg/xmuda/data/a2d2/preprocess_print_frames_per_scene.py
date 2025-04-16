def print_frames_per_scene(root_dir):
    from tabulate import tabulate
    import matplotlib.pyplot as plt
    glob_path = osp.join(root_dir, '*')
    seq_paths = sorted(glob.glob(glob_path))
    table = []
    for seq_path in seq_paths:
        if osp.isfile(seq_path):
            continue
        frame_paths = sorted(glob.glob(osp.join(seq_path, 'camera',
            'cam_front_center', '*.png')))
        cur_split = 'N/A'
        for split_name in ['train', 'test']:
            if osp.basename(seq_path) in getattr(splits, split_name):
                cur_split = split_name
                break
        if cur_split != 'N/A':
            table.append([osp.basename(seq_path), cur_split, len(frame_paths)])
        plt.imshow(np.array(Image.open(frame_paths[0])))
        plt.title(osp.basename(seq_path))
        plt.show()
    header = ['Seq', 'Split', '# Frames']
    print_table = tabulate(table, headers=header, tablefmt='psql')
    print(print_table)
