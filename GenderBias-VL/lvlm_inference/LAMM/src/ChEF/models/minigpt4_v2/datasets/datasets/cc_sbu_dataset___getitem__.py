def __getitem__(self, index):
    ann = self.annotation[index]
    img_file = '{}.jpg'.format(ann['image_id'])
    image_path = os.path.join(self.vis_root, img_file)
    image = Image.open(image_path).convert('RGB')
    image = self.vis_processor(image)
    caption = ann['caption']
    return {'image': image, 'answer': caption, 'image_id': self.img_ids[ann
        ['image_id']]}
