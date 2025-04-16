def load_and_transform_image_data_clip(self, image_paths, device):
    if image_paths is None:
        return None
    image_ouputs = []
    for image_path in image_paths:
        if isinstance(image_path, Image.Image):
            image = image_path
        elif os.path.exists(image_path):
            image = Image.open(image_path)
        elif image_path.startswith('http://'):
            image = Image.open(requests.get(image_path, stream=True).raw)
        else:
            print('can not load image: ', image_path)
        image_output = self.visual_preprocess(image).to(device)
        image_ouputs.append(image_output)
    return torch.stack(image_ouputs, dim=0)
