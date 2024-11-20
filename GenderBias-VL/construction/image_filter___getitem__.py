def __getitem__(self, idx):
    img_path = self.data[idx]
    img = Image.open(img_path).convert('RGB')
    nsfw_label = self.is_all_black(img)
    img, _ = self.transform(img, None)
    return self.filenames[idx], img, nsfw_label
