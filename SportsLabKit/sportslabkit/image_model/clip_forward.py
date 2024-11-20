def forward(self, x):
    ims = []
    for _x in x:
        im = Image.fromarray(_x)
        im = im.resize(self.image_size)
        im = self.preprocess(im)
        ims.append(im)
    ims = torch.stack(ims)
    with torch.no_grad():
        image_features = self.model.encode_image(ims)
    return image_features
