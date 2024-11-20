def cache_images_to_disk(self, i):
    """Saves an image as an *.npy file for faster loading."""
    f = self.npy_files[i]
    if not f.exists():
        np.save(f.as_posix(), cv2.imread(self.im_files[i]), allow_pickle=False)
