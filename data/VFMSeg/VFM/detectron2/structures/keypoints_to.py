def to(self, *args: Any, **kwargs: Any) ->'Keypoints':
    return type(self)(self.tensor.to(*args, **kwargs))
