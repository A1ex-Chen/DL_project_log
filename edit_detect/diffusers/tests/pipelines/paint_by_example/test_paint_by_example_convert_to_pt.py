def convert_to_pt(self, image):
    image = np.array(image.convert('RGB'))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image
