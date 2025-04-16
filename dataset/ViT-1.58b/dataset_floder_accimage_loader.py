def accimage_loader(path: str) ->Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)
