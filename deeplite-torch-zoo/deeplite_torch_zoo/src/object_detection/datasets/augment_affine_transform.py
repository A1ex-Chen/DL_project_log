def affine_transform(self, img, border):
    """Center."""
    C = np.eye(3, dtype=np.float32)
    C[0, 2] = -img.shape[1] / 2
    C[1, 2] = -img.shape[0] / 2
    P = np.eye(3, dtype=np.float32)
    P[2, 0] = random.uniform(-self.perspective, self.perspective)
    P[2, 1] = random.uniform(-self.perspective, self.perspective)
    R = np.eye(3, dtype=np.float32)
    a = random.uniform(-self.degrees, self.degrees)
    s = random.uniform(1 - self.scale, 1 + self.scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    S = np.eye(3, dtype=np.float32)
    S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)
    T = np.eye(3, dtype=np.float32)
    T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate
        ) * self.size[0]
    T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate
        ) * self.size[1]
    M = T @ S @ R @ P @ C
    if border[0] != 0 or border[1] != 0 or (M != np.eye(3)).any():
        if self.perspective:
            img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=
                (114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(
                114, 114, 114))
    return img, M, s
