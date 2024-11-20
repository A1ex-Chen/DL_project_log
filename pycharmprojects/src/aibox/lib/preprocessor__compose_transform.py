def _compose_transform(self, is_train_or_eval: bool, resized_width: int,
    resized_height: int, right_pad: int, bottom_pad: int) ->transforms.Compose:
    transform = transforms.Compose([transforms.Resize(size=(resized_height,
        resized_width)), transforms.Pad(padding=(0, 0, right_pad,
        bottom_pad), fill=0)])
    return transform
