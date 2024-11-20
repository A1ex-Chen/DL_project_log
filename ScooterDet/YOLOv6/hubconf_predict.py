def predict(self, img_path):
    img, img_src = process_image(img_path, self.img_size, 32)
    img = img.to(self.device)
    if len(img.shape) == 3:
        img = img[None]
    prediction = self.forward(img, img_src.shape)
    out = {k: v.cpu().numpy() for k, v in prediction.items()}
    out['classes'] = [self.class_names[i] for i in out['labels']]
    return out
