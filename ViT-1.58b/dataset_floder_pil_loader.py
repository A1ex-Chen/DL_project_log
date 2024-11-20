def pil_loader(path: str) ->Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
