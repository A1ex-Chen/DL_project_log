def representative_dataset(ncalib=100):
    for _ in range(ncalib):
        data = np.random.rand(1, imgsz[0], imgsz[1], 3)
        yield [data.astype(np.float32)]
