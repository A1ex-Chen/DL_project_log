def forward(self, images: List[Tensor], targets: Optional[List[Dict[str,
    Tensor]]]=None) ->Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
    image_sizes = [img.shape[-2:] for img in images]
    images = torch.stack(images, dim=0)
    image_sizes_list: List[Tuple[int, int]] = []
    for image_size in image_sizes:
        assert len(image_size) == 2
        image_sizes_list.append((image_size[0], image_size[1]))
    image_list = ImageList(images, image_sizes_list)
    return image_list, targets
