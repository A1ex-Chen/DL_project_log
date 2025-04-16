def save_images(self, images: [torch.Tensor, List[torch.Tensor]],
    file_names: Union[str, List[str]]):
    if isinstance(images, list) and isinstance(file_names, list):
        if len(images) != len(file_names):
            raise ValueError(
                f'The arguement images and file_names should be lists with equal length, not {len(images)} and {len(file_names)}.'
                )
    rev_images = self.tensor2imgs(images=images)
    if isinstance(rev_images, list):
        for i, r_img in enumerate(rev_images):
            utils.save_image(r_img, fp=file_names[i])
    else:
        utils.save_image(rev_images, fp=file_names)
