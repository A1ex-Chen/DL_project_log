def representative_dataset_gen(dataset, ncalib=100):
    for n, (path, img, im0s, vid_cap, string) in enumerate(dataset):
        im = np.transpose(img, [1, 2, 0])
        im = np.expand_dims(im, axis=0).astype(np.float32)
        im /= 255
        yield [im]
        if n >= ncalib:
            break
