def __getitem__(self, item):
    smile = self.df.iloc[item, 0]
    smile_len = len(str(smile))
    image = self.make_image(smile)
    return 0, transforms.ToTensor()(image), smile_len
