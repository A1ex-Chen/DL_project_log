def __getitem__(self, index):
    ann = self.annotation[index]
    image_path = os.path.join(self.vis_root, ann['image'])
    image = Image.open(image_path).convert('RGB')
    image = self.vis_processor(image)
    return {'image': image, 'image_id': ann['image_id'], 'instance_id': ann
        ['instance_id']}
