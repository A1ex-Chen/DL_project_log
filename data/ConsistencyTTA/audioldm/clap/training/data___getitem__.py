def __getitem__(self, idx):
    images = self.transforms(Image.open(str(self.images[idx])))
    texts = tokenize([str(self.captions[idx])])[0]
    return images, texts
