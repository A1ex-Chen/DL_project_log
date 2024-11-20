def fov(self):
    return torch.from_numpy(np.array([self.x_fov, self.y_fov], dtype=np.
        float32))
