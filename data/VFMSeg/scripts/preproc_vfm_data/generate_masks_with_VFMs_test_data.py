def test_data():
    test_pkl_dir = (
        '/Labs/Scripts/3DPC/Datasets/3DPC/XMUDA_WITH_VFM/virtual_kitti_preprocess_vfm/train'
        )
    num_a = np.random.randint(0, 1000, 1, dtype='int')
    num_b = np.random.randint(0, 1000, 1, dtype='int')
    pkl_file_path_a = test_pkl_dir + '/' + str(num_a[0]) + '.pkl'
    pkl_file_path_b = test_pkl_dir + '/' + str(num_b[0]) + '.pkl'
    pkl_data_a = []
    pkl_data_b = []
    with open(pkl_file_path_a, 'rb') as f:
        pkl_data_a.extend(pickle.load(f))
    with open(pkl_file_path_b, 'rb') as f:
        pkl_data_b.extend(pickle.load(f))
    print('pkl a len = ', len(pkl_data_a))
    print('pkl b len = ', len(pkl_data_b))
    print('pkl a.sam len = ', len(pkl_data_a[0]['sam']))
    print('pkl b.sam len = ', len(pkl_data_b[0]['sam']))
    c = np.random.randint(0, len(pkl_data_a['seem']), 1, dtype='int')
    print('pkl a.seem max = ', max(pkl_data_a[0]['seem'][c]))
    print('pkl b.seem max = ', max(pkl_data_b[0]['seem'][c]))
    return
