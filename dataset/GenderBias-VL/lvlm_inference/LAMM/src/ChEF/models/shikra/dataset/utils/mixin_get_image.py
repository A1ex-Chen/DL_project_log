def get_image(self, image_path):
    if self.image_folder is not None:
        image_path = os.path.join(self.image_folder, image_path)
    image = read_img_general(image_path)
    return image
