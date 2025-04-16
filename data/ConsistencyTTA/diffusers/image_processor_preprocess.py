def preprocess(self, image: Union[torch.FloatTensor, PIL.Image.Image, np.
    ndarray]) ->torch.Tensor:
    """
        Preprocess the image input, accepted formats are PIL images, numpy arrays or pytorch tensors"
        """
    supported_formats = PIL.Image.Image, np.ndarray, torch.Tensor
    if isinstance(image, supported_formats):
        image = [image]
    elif not (isinstance(image, list) and all(isinstance(i,
        supported_formats) for i in image)):
        raise ValueError(
            f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )
    if isinstance(image[0], PIL.Image.Image):
        if self.do_resize:
            image = [self.resize(i) for i in image]
        image = [(np.array(i).astype(np.float32) / 255.0) for i in image]
        image = np.stack(image, axis=0)
        image = self.numpy_to_pt(image)
    elif isinstance(image[0], np.ndarray):
        image = np.concatenate(image, axis=0) if image[0
            ].ndim == 4 else np.stack(image, axis=0)
        image = self.numpy_to_pt(image)
        _, _, height, width = image.shape
        if self.do_resize and (height % self.vae_scale_factor != 0 or width %
            self.vae_scale_factor != 0):
            raise ValueError(
                f'Currently we only support resizing for PIL image - please resize your numpy array to be divisible by {self.vae_scale_factor}currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor'
                )
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, axis=0) if image[0
            ].ndim == 4 else torch.stack(image, axis=0)
        _, _, height, width = image.shape
        if self.do_resize and (height % self.vae_scale_factor != 0 or width %
            self.vae_scale_factor != 0):
            raise ValueError(
                f'Currently we only support resizing for PIL image - please resize your pytorch tensor to be divisible by {self.vae_scale_factor}currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize option in VAEImageProcessor'
                )
    do_normalize = self.do_normalize
    if image.min() < 0:
        warnings.warn(
            f'Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] when passing as pytorch tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]'
            , FutureWarning)
        do_normalize = False
    if do_normalize:
        image = self.normalize(image)
    return image
