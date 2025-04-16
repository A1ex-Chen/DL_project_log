def cache_images(self):
    """Cache images to memory or disk."""
    b, gb = 0, 1 << 30
    fcn, storage = (self.cache_images_to_disk, 'Disk'
        ) if self.cache == 'disk' else (self.load_image, 'RAM')
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(fcn, range(self.ni))
        pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
        for i, x in pbar:
            if self.cache == 'disk':
                b += self.npy_files[i].stat().st_size
            else:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                b += self.ims[i].nbytes
            pbar.desc = (
                f'{self.prefix}Caching images ({b / gb:.1f}GB {storage})')
        pbar.close()
