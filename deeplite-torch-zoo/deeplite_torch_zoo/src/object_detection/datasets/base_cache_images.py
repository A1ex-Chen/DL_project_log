def cache_images(self, cache):
    """Cache images to memory or disk."""
    b, gb = 0, 1 << 30
    fcn = self.cache_images_to_disk if cache == 'disk' else self.load_image
    with ThreadPool(NUM_THREADS) as pool:
        results = pool.imap(fcn, range(self.ni))
        pbar = tqdm(enumerate(results), total=self.ni, bar_format=
            TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
        for i, x in pbar:
            if cache == 'disk':
                b += self.npy_files[i].stat().st_size
            else:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = x
                b += self.ims[i].nbytes
            pbar.desc = f'{self.prefix}Caching images ({b / gb:.1f}GB {cache})'
        pbar.close()
