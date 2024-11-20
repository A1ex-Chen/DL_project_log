def tmp_ds():
    ds = get_dataset(root='fake_images/celeba_hq_256_ddpm', size=64)
    print(f'ds[0:5] Len: {len(ds[0:5])}, len(ds[0:5][0]): {len(ds[0:5][0])}')
    print(
        f'ds.to_tensor() Len: {len(ds.to_tensor())}, ds.to_tensor()[0].shape: {ds.to_tensor()[0].shape}, ds.to_tensor()[1].shape: {ds.to_tensor()[1].shape}, ds.to_tensor()[2].shape: {ds.to_tensor()[2].shape}'
        )
    print(f'Max: {ds[0][0].max()}, Min: {ds[0][0].min()}')
    ds.save_images(ds[0][0], 'test.jpg')
    ds.save_images([ds[0:2][0][0], ds[0:2][1][0]], ['test.jpg', 'test1.jpg'])
