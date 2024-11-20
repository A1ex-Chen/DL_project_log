def cache_images(self, num_imgs=None):
    assert num_imgs is not None, 'num_imgs must be specified as the size of the dataset'
    mem = psutil.virtual_memory()
    mem_required = self.cal_cache_occupy(num_imgs)
    gb = 1 << 30
    if mem_required > mem.available:
        self.cache_ram = False
        LOGGER.warning('Not enough RAM to cache images, caching is disabled.')
    else:
        LOGGER.warning(
            f'{mem_required / gb:.1f}GB RAM required, {mem.available / gb:.1f}/{mem.total / gb:.1f}GB RAM available, Since the first thing we do is cache, there is no guarantee that the remaining memory space is sufficient'
            )
    print(f'self.imgs: {len(self.imgs)}')
    LOGGER.info('You are using cached images in RAM to accelerate training!')
    LOGGER.info('Caching images...\nThis might take some time for your dataset'
        )
    num_threads = min(16, max(1, os.cpu_count() - 1))
    load_imgs = ThreadPool(num_threads).imap(self.load_image, range(num_imgs))
    pbar = tqdm(enumerate(load_imgs), total=num_imgs, disable=self.rank > 0)
    for i, (x, (h0, w0), shape) in pbar:
        self.imgs[i], self.imgs_hw0[i], self.imgs_hw[i] = x, (h0, w0), shape
