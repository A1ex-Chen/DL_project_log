def __call__(self, frames):
    assert frames.shape[-1
        ] == 3, 'Expecting last dimension for 3-channels RGB (b, h, w, c).'
    if self.tensor_in_tensor_out:
        frames = frames.numpy().astype(np.uint8)
    num_frames = frames.shape[0]
    ops = num_frames * [self.get_random_ops()]
    apply_or_not = num_frames * [np.random.random(size=self.N) > self.p]
    frames = torch.stack(list(map(self._aug, frames, ops, apply_or_not)), dim=0
        ).float()
    return frames
