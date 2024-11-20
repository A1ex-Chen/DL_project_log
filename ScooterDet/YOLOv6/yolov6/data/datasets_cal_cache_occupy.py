def cal_cache_occupy(self, num_imgs):
    """estimate the memory required to cache images in RAM.
        """
    cache_bytes = 0
    num_imgs = len(self.img_paths)
    num_samples = min(num_imgs, 32)
    for _ in range(num_samples):
        img, _, _ = self.load_image(index=random.randint(0, len(self.
            img_paths) - 1))
        cache_bytes += img.nbytes
    mem_required = cache_bytes * num_imgs / num_samples
    return mem_required
